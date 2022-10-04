import 'package:mlpdart/mlp/neuron.dart';
import 'package:mlpdart/math/random_weights.dart' as rw;

class Layer {
  List<Neuron> neurons = [];

  // Constructor for input layers
  Layer(List<double> inputs) {
    for (var value in inputs) {
      neurons.add(Neuron.entry(value, <double>[]));
    }
  }

  // Constructor for hidden & output layers
  Layer.hidden(int neuronsLen, int neuronsWeight) {
    for (int i = 0; i < neuronsLen; i++) {
      List<double> weights = [];
      for (var j = 0; j < neuronsWeight; j++) {
        weights.add(rw.randomWeight(Neuron.minWeight, Neuron.maxWeight));
      }
      double bias = rw.randomWeight(0, 1);
      neurons.add(Neuron.hidden(weights, bias));
    }
  }
}
