class Pair {
  List<double> inputData = [];
  List<double> outputData = [];

  Pair(this.inputData, this.outputData);
}

class Dataset {
  List<Pair> pairs = [];

  // @override
  int getLength() {
    // Lenght of the data
    return pairs.length;
  }
}
