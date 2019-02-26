#define BENCHMARK "OSU MPI%s Bandwidth Pipelined Test"
/*
 * Copyright (C) 2002-2018 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util.h>

#define CHUNK_MIN  (1<<15)
#define CHUNK_MAX  (1<<19)

int
main (int argc, char *argv[])
{
    int myid, numprocs, i, j;
    int size;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t = 0.0;
    int window_size;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = BW;

    set_header(HEADER);
    set_benchmark_name("osu_bw_pipe");

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    po_ret = process_options(argc, argv);

    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_ROCM_NOT_AVAIL:
                fprintf(stderr, "ROCM support not enabled.  Please recompile "
                        "benchmark with ROCM support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_ROCM_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if (numprocs != 2) {
        if (0 == myid) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if (ROCM != options.accel) {
        if (0 == myid) {
            fprintf(stderr, "This test only supports ROCM\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    } else {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    print_header(myid, BW);

    /* Bandwidth test */
    for (int chunk = CHUNK_MIN; chunk <= CHUNK_MAX; chunk *= 2) {
        void *m_buf_1, *m_buf_2;
        HIP_CHECK(hipHostMalloc(&m_buf_1, chunk));
        HIP_CHECK(hipHostMalloc(&m_buf_2, chunk));

        if (0 == myid) printf("chunk size: %d\n", chunk);
        for (size = options.min_message_size; size <= options.max_message_size; size += CHUNK_MAX) {
            window_size = options.window_size;
            set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
            set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

            if (0 == myid) {
                for (i = 0; i < options.iterations + options.skip; i++) {
                    if (i == options.skip) {
                        t_start = MPI_Wtime();
                    }

                    if (size > chunk) {
                        int nstages, k;
                        char *h, *d;

                        nstages = size / chunk;
                        h = (char *)m_buf_1;
                        d = (char *)s_buf;
                        HIP_CHECK(hipMemcpy(h, d, chunk, hipMemcpyDeviceToHost));
                        for (k = 0; k < nstages - 1; k++) {
                            MPI_CHECK(MPI_Isend(h, chunk, MPI_CHAR, 1, 100 + k, MPI_COMM_WORLD, request + k));
                            h = (h == m_buf_1) ? (char *)m_buf_2 : (char *)m_buf_1;
                            d += chunk;
                            HIP_CHECK(hipMemcpy(h, d, chunk, hipMemcpyDeviceToHost));
                        }
                        MPI_CHECK(MPI_Isend(h, chunk, MPI_CHAR, 1, 100 + k, MPI_COMM_WORLD, request + k));
                        MPI_CHECK(MPI_Waitall(nstages, request, reqstat));
                    } else {
                        for (j = 0; j < window_size; j++) {
                            MPI_CHECK(MPI_Isend(s_buf, size, MPI_CHAR, 1, 100 + j, MPI_COMM_WORLD, request + j));
                        }
                        MPI_CHECK(MPI_Waitall(window_size, request, reqstat));
                    }
                    MPI_CHECK(MPI_Recv(r_buf, 4, MPI_CHAR, 1, 10, MPI_COMM_WORLD, &reqstat[0]));
                }
                t_end = MPI_Wtime();
                t = t_end - t_start;
            }
            else if (1 == myid) {
                for (i = 0; i < options.iterations + options.skip; i++) {
                    if (size > chunk) {
                        int nstages, k;
                        char *d;

                        nstages = size / chunk;
                        d = (char *)r_buf;

                        for (k = 0; k < nstages; k++) {
                            MPI_CHECK(MPI_Irecv(d, chunk, MPI_CHAR, 0, 100 + k, MPI_COMM_WORLD, request + k));
                            d += chunk;
                        }
                        MPI_CHECK(MPI_Waitall(nstages, request, reqstat));
                    } else {
                        for(j = 0; j < window_size; j++) {
                            MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, 0, 100 + j, MPI_COMM_WORLD, request + j));
                        }
                        MPI_CHECK(MPI_Waitall(window_size, request, reqstat));
                    }
                    MPI_CHECK(MPI_Send(s_buf, 4, MPI_CHAR, 0, 10, MPI_COMM_WORLD));
                }
            }

            if (0 == myid) {
                double tmp;

                if (size > chunk) window_size = 1;
                tmp = size / 1e6 * options.iterations * window_size;

                fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION, tmp / t);
                fflush(stdout);
            }
        }

        HIP_CHECK(hipHostFree(m_buf_1));
        HIP_CHECK(hipHostFree(m_buf_2));
    }

    free_memory(s_buf, r_buf, myid);

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Finalize());
    return EXIT_SUCCESS;
}

