#include <benchmark/benchmark.h>  //google benchmark
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_opadd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int slice = 8;

/* Exercise 4.1 */
void addArray(int n, double a[], const double b[], const double c[])
{
    a[0:n] = b[0:n] + c[0:n];
}
void addArrayPar(int n, double a[], const double b[], const double c[]) {
    cilk_for(int i = 0; i < n; i += slice) {
    int from = i, to = std::min(i + slice, n), len = to - from;
    a [from:len] = b [from:len] + c [from:len];
  }
}

void addArraySimd(int n, double a[], const double b[], const double c[]) {
#pragma simd
  cilk_for(int i = 0; i < n; ++i) a[i] = b[i] + c[i];
}

/* Exercise 4.2 */
double sumParReduce(int n, const double a[]) {
  cilk::reducer<cilk::op_add<double>> sum(0);
  cilk_for(int i = 0; i < n; i += slice) {
    int from = i, to = std::min(i + slice, n), len = to - from;
    *sum += __sec_reduce_add(a [from:len]);
  }
  return sum.get_value();
}

double sumParSimd(int n, const double a[]) {
  cilk::reducer<cilk::op_add<double>> sum(0);
#pragma simd
  cilk_for(int i = 0; i < n; ++i)* sum += a[i];
  return sum.get_value();
}

double sumParReduceHierarchical(int n, const double a[]) {
  std::vector<double> partialSums(__cilkrts_get_nworkers());
  cilk_for(int i = 0; i < n; i += slice) {
    int from = i, to = std::min(i + slice, n), len = to - from;
    partialSums.at(__cilkrts_get_worker_number()) +=
        __sec_reduce_add(a [from:len]);
  }
  double result = 0;
  for (int j = 0; j < __cilkrts_get_nworkers(); ++j)
    result += partialSums[j];
  return result;
}

/* Bonus 4.3 */

void addArraysVecPar(int n, double a[], int m, const double b[]) {
  cilk_for(int i = 0; i < n; ++i) {
    int from = i, to = i + n * m;
    a[i] = __sec_reduce_add(b [from:m:n]);
  }
}

void addArraysParVec(int n, double a[], int m, const double b[]) {
  std::vector<double> partialSums(n * __cilkrts_get_nworkers());
  cilk_for(int j = 0; j < m; ++j) {
    auto workerNumber = __cilkrts_get_worker_number();
    double* workerData = &partialSums[workerNumber * n];
#pragma simd
    for (int i = 0; i < n; ++i)
      workerData[i] += b[j * n + i];
  }
  a [0:n] = 0;
  for (int j = 0; j < __cilkrts_get_nworkers(); ++j)
    a [0:n] += (&partialSums[j * n])[0:n];
}

void addArraysMerged(int n, double a[], int m, const double b[]) {
  std::vector<double> partialSums(n * __cilkrts_get_nworkers());

  int mn = m * n;

#pragma simd
  cilk_for(int k = 0; k < mn; ++k) {
    int i = k % n;
    int j = k / n;
    auto workerNumber = __cilkrts_get_worker_number();
    partialSums[workerNumber * n + i] += b[j * n + i];
  }
  a [0:n] = 0;
  for (int j = 0; j < __cilkrts_get_nworkers(); ++j)
    a [0:n] += (&partialSums[j * n])[0:n];
}

struct ArrayData {
  std::vector<double> a, b, c;

  ArrayData(const ArrayData&) = delete;
  ArrayData(ArrayData&&) = delete;

  ArrayData(unsigned aSize, unsigned bSize = 0, unsigned cSize = 0)
      : a(aSize), b(bSize), c(cSize) {
    for (unsigned i = 0; i < bSize; ++i)
      b[i] = i;
    for (unsigned i = 0; i < cSize; ++i)
      c[i] = 1 + i;
  }
};

bool floatEquals(double lhs, double rhs, double epsilon = 1e-5) {
  return std::abs(lhs - rhs) < epsilon;
}

void verifyAddArray(int n,
                    const double a[],
                    const double b[],
                    const double c[]) {
  for (int i = 0; i < n; ++i)
    if (!floatEquals(a[i], b[i] + c[i])) {
      std::printf("i=%d, expected=%g, actual=%g\t", i, b[i] + c[i], a[i]);
      throw std::runtime_error("wrong results");
    }
}
void verifySum(int n, double a[], double sum) {
  double mySum = 0.0;
  for (int i = 0; i < n; ++i)
    mySum += a[i];
  if (!floatEquals(sum, mySum)) {
    std::printf("expected=%g, actual=%g\t", mySum, sum);
    throw std::runtime_error("wrong results");
  }
}
void verifyAddArrays(int n, double a[], int m, const double b[]) {
  for (int i = 0; i < n; ++i) {
    double sum = 0;
    for (int j = 0; j < m; ++j)
      sum += b[j * n + i];
    if (!floatEquals(a[i], sum)) {
      std::printf("i=%d, expected=%g, actual=%g\t", i, sum, a[i]);
      throw std::runtime_error("wrong results");
    }
  }
}


//--------------------------
// Google Benchmark Template
//--------------------------
static void Ex04Arguments(benchmark::internal::Benchmark* b) {
  const auto lowerLimit = 1;
  const auto upperLimit = 28;
  // Generate 
  for(auto i = lowerLimit; i <= upperLimit; ++i)
	  b->Args({1 << i});
}

static void BonusArguments(benchmark::internal::Benchmark* b) {
  const auto lowerLimit = 1;
  const auto upperLimit = 15;
  const auto startM = 4;
  const auto endM = 16;
  // Generate 
  for(auto q = startM; q <= endM; q *= 4)
	  for(auto i = lowerLimit; i <= upperLimit; ++i)
		  b->Args({1 << i,q});
}

static void benchAddArray(benchmark::State& state){
	ArrayData data(state.range(0), state.range(0), state.range(0));
	for(auto _ : state){
		addArray(data.a.size(),data.a.data(),data.b.data(),data.c.data());
		benchmark::DoNotOptimize(data.a.data());
		benchmark::ClobberMemory();
	}
	verifyAddArray(data.a.size(),data.a.data(),data.b.data(),data.c.data());
}

static void benchAddArrayPar(benchmark::State& state){
	ArrayData data(state.range(0), state.range(0), state.range(0));
	for(auto _ : state){
		addArrayPar(data.a.size(),data.a.data(),data.b.data(),data.c.data());
		benchmark::DoNotOptimize(data.a.data());
		benchmark::ClobberMemory();
	}
	verifyAddArray(data.a.size(),data.a.data(),data.b.data(),data.c.data());
}

static void benchAddArraySimd(benchmark::State& state){
	ArrayData data(state.range(0), state.range(0), state.range(0));
	for(auto _ : state){
		addArraySimd(data.a.size(),data.a.data(),data.b.data(),data.c.data());
		benchmark::DoNotOptimize(data.a.data());
		benchmark::ClobberMemory();
	}
	verifyAddArray(data.a.size(),data.a.data(),data.b.data(),data.c.data());
}

static void benchSumParReduce(benchmark::State& state){
	ArrayData data(state.range(0));
	double sum;
	for(auto _ : state){
		sum = sumParReduce(data.a.size(),data.a.data());
		benchmark::DoNotOptimize(sum);
		benchmark::ClobberMemory();
	}
	verifySum(data.a.size(),data.a.data(),sum);
}

static void benchSumParSimd(benchmark::State& state){
	ArrayData data(state.range(0));
	double sum;
	for(auto _ : state){
		sum = sumParSimd(data.a.size(),data.a.data());
		benchmark::DoNotOptimize(sum);
		benchmark::ClobberMemory();
	}
	verifySum(data.a.size(),data.a.data(),sum);
}

static void benchSumParReduceHierarchical(benchmark::State& state){
	ArrayData data(state.range(0));
	double sum;
	for(auto _ : state){
		sum = sumParReduceHierarchical(data.a.size(),data.a.data());
		benchmark::DoNotOptimize(sum);
		benchmark::ClobberMemory();
	}
	verifySum(data.a.size(),data.a.data(),sum);
}

static void benchAddArraysVecPar(benchmark::State& state){
	ArrayData data(state.range(0),state.range(0) * state.range(1));
	for(auto _ : state){
		addArraysVecPar(data.a.size(),data.a.data(),state.range(1),data.b.data());
		benchmark::DoNotOptimize(data.a.data());
		benchmark::ClobberMemory();
	}
	verifyAddArrays(data.a.size(),data.a.data(),state.range(1),data.b.data());
}

static void benchAddArraysParVec(benchmark::State& state){
	ArrayData data(state.range(0),state.range(0) * state.range(1));
	for(auto _ : state){
		addArraysParVec(data.a.size(),data.a.data(),state.range(1),data.b.data());
		benchmark::DoNotOptimize(data.a.data());
		benchmark::ClobberMemory();
	}
	verifyAddArrays(data.a.size(),data.a.data(),state.range(1),data.b.data());
}

static void benchAddArraysMerged(benchmark::State& state){
	ArrayData data(state.range(0),state.range(0) * state.range(1));
	for(auto _ : state){
		addArraysMerged(data.a.size(),data.a.data(),state.range(1),data.b.data());
		benchmark::DoNotOptimize(data.a.data());
		benchmark::ClobberMemory();
	}
	verifyAddArrays(data.a.size(),data.a.data(),state.range(1),data.b.data());
}

BENCHMARK(benchAddArray)->Apply(Ex04Arguments)->UseRealTime();
BENCHMARK(benchAddArrayPar)->Apply(Ex04Arguments)->UseRealTime();
BENCHMARK(benchAddArraySimd)->Apply(Ex04Arguments)->UseRealTime();
BENCHMARK(benchSumParReduce)->Apply(Ex04Arguments)->UseRealTime();
BENCHMARK(benchSumParSimd)->Apply(Ex04Arguments)->UseRealTime();
BENCHMARK(benchSumParReduceHierarchical)->Apply(Ex04Arguments)->UseRealTime();
BENCHMARK(benchAddArraysVecPar)->Apply(BonusArguments)->UseRealTime();
BENCHMARK(benchAddArraysParVec)->Apply(BonusArguments)->UseRealTime();
BENCHMARK(benchAddArraysMerged)->Apply(BonusArguments)->UseRealTime();
BENCHMARK_MAIN();
