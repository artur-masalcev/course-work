# course-work


## Introduction
Repository for course work on the topic of "Parallel sorting algorithms".
In `odd-even_sort.cpp` algorithm for Odd-Even Sort is stored. And `sample_sort.cpp` contains the code for Sample Sort algorithm.

## Prerequisites
- An MPI implementation (like MPICH or OpenMPI)
- C++ compiler supporting C++11 or higher
- The source code file (provided)

## Compilation
To compile the program, use the following command:
```bash
mpicxx -o parallel_sort parallel_sort.cpp
```

## Running the Program

To run the program, use the mpiexec or mpirun command with the desired number of processes. For example:

```bash
mpiexec -n 4 ./parallel_sort
```

This command runs the program with 4 processes.
