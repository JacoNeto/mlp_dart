/// classe [Neuron] se refere aos neurônios do MLP,
/// contendo atributos com seu valor [value], seus pesos [weights],
/// o viés [bias] e o gradiente [gradient]

class Neuron {
  // o valor pode ser null, já que apenas neurônios de entrada possuem valores
  // nos construtores
  double? value;
  List<double> weights = [];
  List<double> previousWeights = [];
  double bias = 0.0;
  double gradient = 0.0;

  // construtor para neurônios de entrada
  Neuron.entry(this.value, this.weights) {
    bias = -1.0;
    gradient = -1.0;
    previousWeights = weights;
  }

  // construtor para neurônios ocultos e de saída
  Neuron.hidden(this.weights, this.bias) {
    previousWeights = weights;
    gradient = 0.0;
  }

  // utilizado para definir e atualizar os pesos dos neurônios
  setWeights() {
    weights = previousWeights;
  }

  /// Intervalo dos pesos. Por padrão é 0 e 1, mas é possível customizar isso
  /// chamando a função [setRange]
  static int minWeight = 0;
  static int maxWeight = 1;
  static void setRange(int min, int max) {
    minWeight = min;
    maxWeight = max;
  }
}
