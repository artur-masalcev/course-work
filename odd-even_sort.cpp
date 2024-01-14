#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

const int SIZE = 250000;
const bool DEBUG_MESSAGES_ENABLED = false;

double COMM_TIME = 0;

void generateRandomArray(int* arr, int size, int rank) {
    srand(time(0) + rank);

    for (int i = 0; i < size; ++i) {
        int randomNumber = rand() % 999999; // Generate a random number between 0 and 999999
        arr[i] = randomNumber;
    }
}

void add_comm_time(double start_time, double end_time)
{
    COMM_TIME += end_time - start_time;
}

void printNumbers(int arr[], int size, int rank) {
    if(!DEBUG_MESSAGES_ENABLED) return;

    cout << "Process " << rank << "-> ";
    cout << "Numbers: ";
    
    for (int i = 0; i < size; ++i) {
        cout << arr[i] << " ";
    }
    
    cout << endl;
}

void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void distribute_numbers(int comm_size) {
    // Generate a set of random numbers for each child node
    for (int i = 1; i < comm_size; ++i) {
        int* numbers = new int[SIZE];
        
        generateRandomArray(numbers, SIZE, i);
        
        MPI_Send(numbers, SIZE, MPI_INT, i, 0, MPI_COMM_WORLD);
        
        delete(numbers);
    }
}

void merge_asc(int a1[], int a2[], int size1, int size2) {
    int* temp = new int[size1];

    // Clone a1 to temp
    for (int i = 0; i < size1; i++) {
        temp[i] = a1[i];
    }

    int temp_ptr = 0, a2_ptr = 0, a1_ptr = 0;

    // Traverse temp and a2 and fill a1
    while (temp_ptr < size1 && a2_ptr < size2 && a1_ptr < size1) {
        if (temp[temp_ptr] <= a2[a2_ptr]) {
            a1[a1_ptr++] = temp[temp_ptr++];
        } 
        else {
            a1[a1_ptr++] = a2[a2_ptr++];
        }
    }

    // If there are remaining elements in temp, add them to a1
    while (temp_ptr < size1 && a1_ptr < size1) {
        a1[a1_ptr++] = temp[temp_ptr++];
    }

    // If there are remaining elements in a2, add them to a1
    while (a2_ptr < size2 && a1_ptr < size1) {
        a1[a1_ptr++] = a2[a2_ptr++];
    }

    delete[] temp;
}

void merge_desc(int a1[], int a2[], int size1, int size2) {
    int* temp = new int[size1];

    // Clone a1 to temp
    for (int i = 0; i < size1; i++) {
        temp[i] = a1[i];
    }

    int temp_ptr = size1 - 1;
    int a2_ptr = size2 - 1;
    int a1_ptr = size1 - 1;

    // Traverse temp and a2 and fill a1
    while (temp_ptr >= 0 && a2_ptr >= 0 && a1_ptr >= 0) {
        if(temp[temp_ptr] >= a2[a2_ptr]) {
            a1[a1_ptr--] = temp[temp_ptr--];
        }
        else {
            a1[a1_ptr--] = a2[a2_ptr--];
        }
    }

    // Fill the rest of a1 with the remaining values from temp
    while (temp_ptr >= 0 && a1_ptr >= 0) {
        a1[a1_ptr--] = temp[temp_ptr--];
    }

    // Fill the rest of a1 with the remaining values from a2
    while (a2_ptr >= 0 && a1_ptr >= 0) {
        a1[a1_ptr--] = a2[a2_ptr--];
    }

    delete[] temp;
}

void even_phase(int procnum, int even_partner, int* numbers) {
    MPI_Status status;

    if (even_partner == MPI_PROC_NULL) return;

    int* received_numbers = new int[SIZE];

    double comm_start_time = MPI_Wtime();
    MPI_Sendrecv(numbers, SIZE, MPI_INT, even_partner, 0,
        received_numbers, SIZE, MPI_INT, even_partner, 0, MPI_COMM_WORLD,
        &status);
    double comm_end_time = MPI_Wtime();
    add_comm_time(comm_start_time, comm_end_time);
    
    if(procnum % 2 != 0) {
        merge_desc(numbers, received_numbers, SIZE, SIZE);
    }
    else {
        merge_asc(numbers, received_numbers, SIZE, SIZE);
    }

    delete(received_numbers);
}

void odd_phase(int procnum, int odd_partner, int* numbers) {
    MPI_Status status;

    if (odd_partner == MPI_PROC_NULL) return;

    int* received_numbers = new int[SIZE];

    double comm_start_time = MPI_Wtime();
    MPI_Sendrecv(numbers, SIZE, MPI_INT, odd_partner, 0,
        received_numbers, SIZE, MPI_INT, odd_partner, 0, MPI_COMM_WORLD,
        &status);
    double comm_end_time = MPI_Wtime();
    add_comm_time(comm_start_time, comm_end_time);

    if(procnum % 2 != 0) {
        merge_asc(numbers, received_numbers, SIZE, SIZE);
    }
    else {
        merge_desc(numbers, received_numbers, SIZE, SIZE);
    }

    delete(received_numbers);
}

void collect_sorted_data(int* result_array, int* self_numbers, int comm_size) {
    // First insert into result_array numbers from node 0
    for (int i = 0; i < SIZE; ++i) {
        result_array[i] = self_numbers[i];
    }
    
    // Receive and insert numbers from other nodes
    for (int i = 1; i < comm_size; ++i) {
        int* numbers = new int[SIZE];

        MPI_Status status;
        MPI_Recv(numbers, SIZE, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

        int start_index = i * SIZE;

        for(int j = start_index; j < start_index + SIZE; ++j) {
            result_array[j] = numbers[j - start_index];
        }

        delete numbers;
    }
}

int main(int argc, char *argv[]) {
    int procnum, comm_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &procnum);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int* numbers = new int[SIZE];
    generateRandomArray(numbers, SIZE, procnum);

    printNumbers(numbers, SIZE, procnum);

    barrier();

    double start_time = MPI_Wtime();

    // Local sort
    sort(numbers, numbers + SIZE);

    barrier();

    // Determine neighbour index for each phase
    int even_partner;
    int odd_partner;
    
    if (procnum % 2 != 0) {
        even_partner = procnum - 1;
        odd_partner = (procnum == comm_size - 1) ? MPI_PROC_NULL : (procnum + 1);
    }
    else {
        even_partner = (procnum == comm_size - 1) ? MPI_PROC_NULL : (procnum + 1);
        odd_partner = (procnum == 0) ? MPI_PROC_NULL : procnum - 1;
    }
    
    // Odd-Even phases
    for(int phase = 0; phase < comm_size; phase++) {
        if(phase % 2 == 0) {
            even_phase(procnum, even_partner, numbers);
        }
        else {
            odd_phase(procnum, odd_partner, numbers);
        }
        barrier();
    }

    double end_time = MPI_Wtime();

    double elapsed_time = end_time - start_time;

    cout << "Process: " << procnum << " Elapsed time: " << elapsed_time << endl;
    cout << "Process: " << procnum << " Communication time: " << COMM_TIME << endl;

    printNumbers(numbers, SIZE, procnum);

    MPI_Finalize();
    
    return 0;
}