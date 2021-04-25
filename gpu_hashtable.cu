#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

/* Hash functions
 */

__device__ uint32_t murmur3_h(uint32_t key)
{
	key ^= key >> 16;
	key *= 0x85ebca6b;
	key ^= key >> 13;
	key *= 0xc2b2ae35;
	key ^= key >> 16;
	return key;
}

__constant__ uint32_t closest_prime;
__constant__ uint32_t gpu_hash_size;

__device__ uint32_t hash_occupation;

__device__ void gpu_insert(KeyValue *dest_hash, KeyValue ins)
{
	uint64_t pos = murmur3_h(ins.key);
	uint32_t increment = closest_prime - (ins.key % closest_prime);
	uint32_t old_key, idx;

	while(true)
	{
		idx = pos % gpu_hash_size;
		old_key = atomicCAS(&dest_hash[idx].key, NULL_KEY, ins.key);
		
		if(old_key == NULL_KEY)
			atomicAdd(&hash_occupation,1);

		if( old_key == NULL_KEY || old_key == ins.key) {
			dest_hash[idx].value = ins.value;
			return;
		}
		pos += increment;
	}
}

__global__ void gpu_insert_batch(KeyValue *dest_hash, uint32_t* keys,
									uint32_t *values, uint32_t size) {
	uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

	if( pos < size) {
		KeyValue k;
		k.key = keys[pos];
		k.value = values[pos];
		gpu_insert(dest_hash,k);
	}
}

__global__ void gpu_rehash(KeyValue* source, KeyValue *destination, 
						uint32_t size_s) {
	uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

	if( pos < size_s && source[pos].key != 0 )
			gpu_insert(destination,source[pos]);
}

__global__ void gpu_get_value(KeyValue* hash, uint32_t *keys, uint32_t keyNum) {
	uint32_t keys_pos = blockIdx.x * blockDim.x + threadIdx.x;

	if( keys_pos >= keyNum)
		return;

	uint32_t key = keys[keys_pos];
	uint64_t pos = murmur3_h(key);
	uint32_t increment = closest_prime - (key % closest_prime);
	uint32_t old_key, idx;

	while(true)
	{
		idx = pos % gpu_hash_size;

		old_key = atomicCAS(&hash[idx].key, key, key);

		if( old_key == NULL_KEY) {
			keys[keys_pos]=0;
			return;
		}
		
		if(old_key == key) {
			keys[keys_pos] = hash[idx].value;
			return;
		}
		pos += increment;
	}
}


/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	cudaMalloc((void **) &hash_table, size * sizeof(KeyValue));
	cudaMemset(hash_table, 0, size * sizeof(KeyValue));
	hash_size = size;
	block_size = 256;
	hash_occupied = 0;
	load_factor = 1.0f;
	cudaMemcpyToSymbol(hash_occupation, &hash_occupied, sizeof(uint32_t) );
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hash_table);
}

/* CHANGE THE CLOSEST PRIME TO HASH SIZE
 */
void GpuHashTable::setPrimeConst(uint32_t number) {
	uint32_t pow, sol;
	for(pow = 1; pow < PRIMES_NUM; pow <<=1);

	for(sol=0; pow ; pow >>=1)
		if( sol + pow < PRIMES_NUM)
			if( primeList[sol+pow] < number)
				sol+=pow;
	
	cudaMemcpyToSymbol(closest_prime, &primeList[sol], sizeof(uint32_t) );
}

uint32_t get_closest_bigger_prime(uint32_t number) {
	int pow, sol;
	for( pow =1; pow < PRIMES_NUM; pow <<=1);

	for(sol=PRIMES_NUM; pow; pow >>=1)
		if( sol-pow >= 0)
			if( primeList[sol-pow] >= number)
				sol-=pow;
	return primeList[sol];
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	KeyValue *aux;
	numBucketsReshape = get_closest_bigger_prime(numBucketsReshape);
	cudaMalloc((void **) &aux, numBucketsReshape*sizeof(KeyValue));
	cudaMemset(aux, 0, numBucketsReshape * sizeof(KeyValue));

	hash_occupied = 0;
	cudaMemcpyToSymbol(hash_occupation, &hash_occupied, sizeof(uint32_t) );
	
	uint32_t block_num = hash_size / block_size;

	if( block_num * block_size < hash_size)
		block_num++;

	setPrimeConst(numBucketsReshape);
	cudaMemcpyToSymbol(gpu_hash_size, &numBucketsReshape, sizeof(uint32_t));

	gpu_rehash<<<block_num, block_size>>>(hash_table, aux, hash_size);
	cudaDeviceSynchronize();
	cudaFree(hash_table);
	hash_table = aux;
	hash_size = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaMemcpyFromSymbol(&hash_occupied,hash_occupation,sizeof(uint32_t),
						0,cudaMemcpyDeviceToHost);
	if(hash_occupied+numKeys > hash_size) {
		reshape((uint32_t)(load_factor*(hash_occupied+numKeys)));
	}
	
	uint32_t block_num = numKeys / block_size;
	if( block_num*block_size < numKeys)
		block_num++;

	uint32_t *device_keys, *device_values;
	cudaMalloc((void **) &device_keys, numKeys*sizeof(uint32_t));
	cudaMalloc((void **) &device_values, numKeys*sizeof(uint32_t));

	cudaMemcpy(device_keys, keys, numKeys*sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys*sizeof(uint32_t), cudaMemcpyHostToDevice);

	gpu_insert_batch<<<block_num, block_size>>>(hash_table, device_keys, device_values, numKeys);
	cudaDeviceSynchronize();
	cudaFree(device_keys);
	cudaFree(device_values);
	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result = (int *) malloc ( numKeys * sizeof(int));
	uint32_t *device_result;

	cudaMalloc((void **) &device_result, numKeys*sizeof(uint32_t));
	cudaMemcpy(device_result, keys, numKeys*sizeof(uint32_t), cudaMemcpyHostToDevice);
	uint32_t block_num = numKeys / block_size;
	if( block_num*block_size < numKeys)
		block_num++;

	gpu_get_value<<<block_num, block_size>>>(hash_table, device_result, numKeys);
	cudaDeviceSynchronize();
	cudaMemcpy(result, device_result, numKeys*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(device_result);

	return result;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	cudaMemcpyFromSymbol(&hash_occupied,hash_occupation,sizeof(uint32_t),
						0,cudaMemcpyDeviceToHost);
	return (float)hash_occupied / hash_size;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
