// nesse arquivo se encontram as principais funções matemáticas utilizadas
// mo MLP

import 'dart:math' as math;

// Função Sigmoide
double sigmoid(double x) {
  var sigmoid = (1 / (1 + math.pow(math.e, -1 * x))) / 1.0;
  return sigmoid;
}

// Derivada da função sigmoide
double sigmoidDerivative(double x) {
  // Valor do sigmoide de x
  var sigmoidV = sigmoid(x);
  // Valor da derivada do sigmoide
  var derivative = sigmoidV * (1 - sigmoidV);
  return derivative;
}

// Erro usado no algoritmo de backpropagation
double squaredError(double output, double target) {
  return (0.5 * math.pow(2, (target - output))).toDouble();
}

// Função utilizada para calcular o erro total da rede
double sumSquaredError(List<double> outputs, List<double> targets) {
  double sum = 0;
  for (int i = 0; i < outputs.length; i++) {
    sum += squaredError(outputs[i], targets[i]);
  }
  return sum;
}
