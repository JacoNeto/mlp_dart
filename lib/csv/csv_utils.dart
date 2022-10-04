import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart';

class XMLUtils {
  static Future<List<List<dynamic>>> loadData() async {
    final input = File('data/newentriesAI.csv').openRead();
    final fields = await input
        .transform(utf8.decoder)
        .transform(CsvToListConverter(eol: '\n', shouldParseNumbers: false))
        .toList();

    return fields;
  }
}
