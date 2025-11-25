/* 4–20 mA Messung mit Arduino Mega 2560 (A0 & A1) über 220 Ω Shunts
- 4 mA -> ~0.88 V
- 20 mA -> ~4.40 V
- ADC: 10 Bit (0..1023)
Verdrahtung pro Kanal:
Sensor– -> Shunt (220 Ω) -> GND (gemeinsam mit Arduino)
A0/A1 -> Knoten Sensor– / Shunt-OBEN (Seite NICHT an GND)
Arduino GND mit Sensor-GND verbinden (gemeinsame Masse!)
*/
//////////////////////
// Konfiguration
//////////////////////
int PIN_ADC_1 = A0; // Mess-Pin Kanal 1
int PIN_ADC_2 = A1; // Mess-Pin Kanal 2
float SHUNT_OHM = 220.0; // Shunt-Widerstand in Ohm
float VREF = 5.00; // Tatsächliche 5V am Board messen (z. B. 4.96) und hier eintragen
unsigned int SAMPLES = 16; // Mittelung über n Messungen
float ALPHA = 0.15; // Glättung (0..1): höher = schneller
// Optional: auf reale Einheit (z. B. l/min) mappen
//////////////////////
// Interne Variablen
//////////////////////
float currentFiltered_mA_1 = 0.0;
float currentFiltered_mA_2 = 0.0;
void setup() {
Serial.begin(115200);
analogReference(DEFAULT); // Mega: nutzt ~5V Boardversorgung
pinMode(PIN_ADC_1, INPUT);
pinMode(PIN_ADC_2, INPUT);
// ADC "beruhigen"
for (int i = 0; i < 8; i++) {
analogRead(PIN_ADC_1);
analogRead(PIN_ADC_2);
}
Serial.println(F("time_ms,chan,raw,voltage_V,current_mA,percent,engineering"));
}
void loop() {
readAndPrintChannel(PIN_ADC_1, currentFiltered_mA_1, 1);
readAndPrintChannel(PIN_ADC_2, currentFiltered_mA_2, 2);
delay(100); // ~10 Hz pro Kanal
}
void readAndPrintChannel(int pin, float &filt_mA, int chan) {
// 1) Mehrfach lesen und mitteln
unsigned long acc = 0;
for (unsigned int i = 0; i < SAMPLES; i++) {
acc += analogRead(pin);
}
float raw = acc / (float)SAMPLES;
// 2) In Spannung umrechnen
float voltage = raw * (VREF / 1023.0);
// 3) In Strom (mA) umrechnen
float current_mA = (voltage / SHUNT_OHM) * 1000.0;
// 4) Sanft glätten (IIR)
filt_mA += ALPHA * (current_mA - filt_mA);
// 5) Prozent (4–20 mA -> 0..100 %)
float percent = (filt_mA - 4.0) / 16.0 * 100.0;
if (percent < 0) percent = 0;
if (percent > 100) percent = 100;
// 6) Optional: Engineering-Unit
// 7) Ausgabe
Serial.print(millis()); Serial.print(',');
Serial.print(chan); Serial.print(',');
Serial.print(raw, 1); Serial.print(',');
Serial.print(voltage, 3);Serial.print(',');
Serial.print(filt_mA, 3);Serial.print(',');
Serial.print(percent, 1);Serial.print(',');
	Serial.println();
}

