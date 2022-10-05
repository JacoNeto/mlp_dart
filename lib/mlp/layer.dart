import 'package:mlpdart/mlp/neuron.dart';
import 'package:mlpdart/math/random_weights.dart' as rw;

/// Uma [Layer] do MLP consiste basicamente numa camada de neurônios [neuron].
/// Cada camada pode ser de entrada, representada pelo construtor [Layer.entry],
/// além de oculta ou de saída, ambas representadas pelo construtor [Layer.entry]

class Layer {
  // Neurônios da camada
  List<Neuron> neurons = [];

  // Construtor para as camadas de entrada
  Layer.entry(List<double> inputs) {
    // recebe uma lista de valores de entrada, inicializa neurônios de entrada
    // com esses valores e adiciona à lista de neurônios da camada
    for (var value in inputs) {
      neurons.add(Neuron.entry(value, <double>[]));
    }
  }

  // Construtor para as camadas de saída
  Layer.hidden(int neuronsLen, int neuronsWeight) {
    // recebe o número de neurônios da camada e quantos pesos cada um vai ter
    for (int i = 0; i < neuronsLen; i++) {
      List<double> weights = [];
      // inicializa os pesos aleatoriamente
      for (var j = 0; j < neuronsWeight; j++) {
        weights.add(rw.randomWeight(Neuron.minWeight, Neuron.maxWeight));
      }
      double bias = rw.randomWeight(0, 1);

      // adiciona o neurônio oculto com o peso e o viés
      neurons.add(Neuron.hidden(weights, bias));
    }
  }
}
