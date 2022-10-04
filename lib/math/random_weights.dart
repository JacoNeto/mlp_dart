import 'dart:math' as math;

/// função que retorna pesos aleatórios num range de [min] e [max]
double randomWeight(int min, int max) {
  // wraper no random do dart
  math.Random random = math.Random();

  // calcula o valor random
  var randomWeight = random.nextDouble() * (max - min) + min;

  return randomWeight;
}
