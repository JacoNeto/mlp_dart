// Um par de dados é um conjunto de entrada e saída
class Pair {
  List<double> inputData = [];
  List<double> outputData = [];

  Pair(this.inputData, this.outputData);
}

// Um dataset é uma lista de pares
class Dataset {
  List<Pair> pairs = [];

  // @override
  int getLength() {
    // Tamanho dos dados
    return pairs.length;
  }
}
