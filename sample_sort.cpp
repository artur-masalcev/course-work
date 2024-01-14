#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

int SIZE = 250000; // Count of numbers to sort in each node
bool DEBUG_MESSAGES_ENABLED = false; // Enable debug messages in console

using namespace std;

void print_numbers(int* numbers, int numbers_size, int procnum, string message) {
    if (!DEBUG_MESSAGES_ENABLED) return;
    cout << "Process " << procnum << " [" << message << "]: ";
    for(int i = 0; i < numbers_size; ++i) {
        cout << numbers[i] << " ";
    }
    cout << endl;
}

void generate_random_array(int* arr, int size, int rank) {
    srand(time(0) + rank);

    for (int i = 0; i < size; ++i) {
        int randomNumber = rand() % 999999; // Generate a random number between 0 and 999999
        arr[i] = randomNumber;
    }
}

int* choose_samples(int* numbers, int numbers_size, int count) {
    int* samples = new int[count];

    int step = numbers_size / (count + 1);

    for(int i = 0; i < count; ++i) {
        samples[i] = numbers[step * (i+1)];
    }

    return samples;
}

void print_time(double start_time, double end_time, int procnum, string message) {
    double elapsed_time = end_time - start_time;
    cout << "Process " << procnum << " [Elapsed time - " << message << "]: ";
    printf("%f seconds\n", elapsed_time);
}

void print_time(double elapsed_time, int procnum, string message) {
    cout << "Process " << procnum << " [Elapsed time - " << message << "]: ";
    printf("%f seconds\n", elapsed_time);
}

double calculate_elapsed_time(double start_time, double end_time) {
    return end_time - start_time;
}

void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    MPI_Status status;

    int procnum, comm_size;

    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &procnum);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int* numbers = new int[SIZE];
    generate_random_array(numbers, SIZE, procnum);

    int samples_size = 100;
    int* samples = choose_samples(numbers, SIZE, samples_size);

    print_numbers(numbers, SIZE, procnum, "Generated numbers");

    barrier();

    // Accept/Send samples to 0th node
    int* all_samples = nullptr;
    
    int* ranges;
    int ranges_size = comm_size-1;

    double samples_sending_start_time = MPI_Wtime();
    if (procnum == 0) {
        int all_samples_size = samples_size * comm_size;
        all_samples = new int[all_samples_size];
        int samples_idx = 0;

        // fill with own samples
        for(int i = 0; i < samples_size; i++) {
            all_samples[samples_idx++] = samples[i];
        }

        // get samples from others
        for (int i = 1; i < comm_size; ++i) { 
            MPI_Recv(samples, samples_size, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

            for(int j = 0; j < samples_size; ++j) {
                all_samples[samples_idx++] = samples[j];
            }
        }

        // sort all samples
        sort(all_samples, all_samples + all_samples_size);

        // select sorting ranges for nodes
        ranges = choose_samples(all_samples, all_samples_size, comm_size - 1);

        MPI_Bcast(ranges, ranges_size, MPI_INT, 0, MPI_COMM_WORLD);

        print_numbers(ranges, ranges_size, procnum, "Generated ranges");
    }
    else {
        MPI_Send(samples, samples_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        // receive ranges from 0th node
        ranges = new int[ranges_size];
        MPI_Bcast(ranges, ranges_size, MPI_INT, 0, MPI_COMM_WORLD);
    }    

    double samples_sending_end_time = MPI_Wtime();
    barrier();
    int bucket_size = comm_size;

    vector<int> *buckets = new vector<int>[bucket_size];

    // fill local buckets
    for (int i = 0; i < SIZE; ++i) {
        // Find the right bucket using binary search
        int bucket_index = std::upper_bound(ranges, ranges + ranges_size, numbers[i]) - ranges;

        // Place the number in the correct bucket
        buckets[bucket_index].push_back(numbers[i]);
    }

    barrier();

    int* received_buckets_sizes = new int[bucket_size];
    vector<int>* received_buckets = new vector<int>[bucket_size];

    // Communicate bucket sizes using MPI_Alltoall
    double bsize_communication_start_time = MPI_Wtime();

    int* all_bucket_sizes = new int[bucket_size]; // Array to store all bucket sizes
    for(int i = 0; i < bucket_size; ++i) {
        all_bucket_sizes[i] = buckets[i].size();
    }
    MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_bucket_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < bucket_size; ++i) {
        received_buckets_sizes[i] = all_bucket_sizes[i];
        received_buckets[i].resize(all_bucket_sizes[i]);
    }
    delete[] all_bucket_sizes;
    delete[] ranges;
    if (all_samples != nullptr) {
        delete[] all_samples;
        all_samples = nullptr;
    }

    double bsize_communication_end_time = MPI_Wtime();

    // Communicate bucket data
    double bdata_sending_start_time = MPI_Wtime();

    int* send_counts = new int[bucket_size];
    int* send_displacements = new int[bucket_size];
    int* recv_counts = new int[bucket_size];
    int* recv_displacements = new int[bucket_size];

    int total_send_count = 0;
    int total_recv_count = 0;
    for (int i = 0; i < bucket_size; ++i) {
        send_counts[i] = buckets[i].size();
        recv_counts[i] = received_buckets_sizes[i];
        send_displacements[i] = total_send_count;
        recv_displacements[i] = total_recv_count;
        total_send_count += send_counts[i];
        total_recv_count += recv_counts[i];
    }

    int* send_buffer = new int[total_send_count];
    int* recv_buffer = new int[total_recv_count];

    for (int i = 0; i < bucket_size; ++i) {
        std::copy(buckets[i].begin(), buckets[i].end(), send_buffer + send_displacements[i]);
    }

    MPI_Alltoallv(send_buffer, send_counts, send_displacements, MPI_INT,
                recv_buffer, recv_counts, recv_displacements, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < bucket_size; ++i) {
        std::copy(recv_buffer + recv_displacements[i], 
                recv_buffer + recv_displacements[i] + recv_counts[i], 
                received_buckets[i].begin());
    }

    delete[] send_counts;
    delete[] send_displacements;
    delete[] recv_counts;
    delete[] recv_displacements;
    delete[] send_buffer;
    delete[] recv_buffer;

    double bdata_sending_end_time = MPI_Wtime();

    barrier();

    // Merge received buckets and sort
    vector<int> merged_buckets;
    size_t total_size = 0;

    // Calculate total size
    for (int i = 0; i < bucket_size; ++i) {
        total_size += received_buckets_sizes[i];
    }

    double bdata_sorting_start_time = MPI_Wtime();
    
    // Reserve space
    merged_buckets.reserve(total_size);

    // Merge vectors
    for (int i = 0; i < bucket_size; ++i) {
        merged_buckets.insert(merged_buckets.end(), received_buckets[i].begin(), received_buckets[i].end());
    }

    // Sort the merged vector
    sort(merged_buckets.begin(), merged_buckets.end());
    double bdata_sorting_end_time = MPI_Wtime();

    print_numbers(merged_buckets.data(), total_size, procnum, "Final set of numbers");

    double end_time = MPI_Wtime();
    
    print_time(start_time, end_time, procnum, "Overall");

    double c1 = calculate_elapsed_time(samples_sending_start_time, samples_sending_end_time);
    double c2 = calculate_elapsed_time(bsize_communication_start_time, bsize_communication_end_time);
    double c3 = calculate_elapsed_time(bdata_sending_start_time, bdata_sending_end_time);

    print_time(c1+c2+c3, procnum, "Overall communication");

    MPI_Finalize();
    return 0;
}