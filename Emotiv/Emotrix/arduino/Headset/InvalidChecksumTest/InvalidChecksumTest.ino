String index, binary, sample, bin1, bin2, chars;
char chr1, chr2;
long valor_random;
int suma;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    // Esperamos a que se conecte el puerto serial
  }

  randomSeed(analogRead(0));
}

String decimal_to_bynary(int value) {
  int   i = 15;
  String binaryStr = "";

  while (i >= 0) {
    if ((value >> i) & 1)
      binaryStr += "1";
    else
      binaryStr += "0";
    i -= 1;
  }

  return binaryStr;
}

int potencia(int a, int b) {
  if (b == 0)
    return 1;

  int result = a;
  for (int i=1; i<b; i++)
    result *= a;

  return result;
}

int bynary_to_decimal(String bynary) {
  int result = 0;
  int pot = bynary.length() - 1;
  for (int i = 0; i<bynary.length(); i++){
    if (bynary.substring(i, i+1) == "1")
      result += potencia(2, pot);
    pot -= 1;
  }

  return result;
}

void loop() {
  // A good sinal value should be between 0 and 4095.
  sample = "{";
  suma = 0;

  for (int i=1; i <5 ; i++) {
    valor_random = random(0, 4096);
    suma += valor_random;

    // Alterate random value adding 1 o 0.
    valor_random += (int) random(0,2);

    binary = decimal_to_bynary(valor_random);
    bin1 = binary.substring(0, 8);
    bin2 = binary.substring(8, 16);
  
    chr1 = bynary_to_decimal(bin1);
    // WARNING: Correct this unresolved problem.    
    if (chr1 == 0){
      chr1 = 1;
      suma += 256;
    }
    chr2 = bynary_to_decimal(bin2);

    index = "s";
    index += i;

    chars = "";
    chars.concat(chr1);
    chars.concat(chr2);

    sample += index + ":" + chars + ",";
  }

  binary = decimal_to_bynary(suma);
  bin1 = binary.substring(0, 8);
  bin2 = binary.substring(8, 16);

  chr1 = bynary_to_decimal(bin1);
  chr2 = bynary_to_decimal(bin2);

  chars = "";
  chars.concat(chr1);
  chars.concat(chr2);

  sample += "cs:" + chars + "}";

  Serial.println(sample);
}
