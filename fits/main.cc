#include "fit_mm.h"
#include <iostream>
#include <cstring>
using namespace std;

int main()
{
  cout << "Hello World!";

  long numel = 10;
  long numte = 10;
  float *phase = new float [numel * numte]; 
  std::memset(phase, 0, sizeof(float) * numel * numel);

  phase[2] = 10;

  unwrap1d(phase, numel, numte);
  
  for (int i = 0; i < numel * numte; ++i){
      cout << phase[i] << "\n";
  }

  return 0;
}