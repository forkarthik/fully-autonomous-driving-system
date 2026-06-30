/* ===============================================
   ADAS LEVEL 3 - ESP32 CONTROL SYSTEM (Jetson Integrated)
   Components: ESP32, L298N, Motors, Sensors
   =============================================== */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ================= PIN DEFINITIONS =================
#define EN_A 25        // Steering PWM
#define IN1 26         // Steering Direction 1
#define IN2 27         // Steering Direction 2
#define EN_B 33        // Drive PWM
#define IN3 14         // Drive Direction 1
#define IN4 12         // Drive Direction 2

// Sensor Pins
#define TRIG_LEFT 5
#define ECHO_LEFT 18
#define TRIG_RIGHT 19
#define ECHO_RIGHT 21
#define RADAR_RX 16
#define RADAR_TX 17
#define POT_PIN 34     // Steering position feedback

// PWM Configuration
#define PWM_FREQ 1000
#define PWM_RES 8
#define DRIVE_CH 0
#define STEER_CH 1

// WiFi Credentials
const char* ssid = "ADAS_CAR";
const char* password = "12345678";

WebServer server(80);

// System State
bool autoMode = false;
int driveSpeed = 0; // -255 to 255
int steerAngle = 0; // -255 to 255 (PWM)
unsigned long lastCmdTime = 0;
const unsigned long TIMEOUT = 1000;

// Serial Parsing
String inputString = "";
bool stringComplete = false;

// Sensor Data
struct SensorData {
  float leftDist;
  float rightDist;
  float radarDist;
  int steerPos;
} sensors;

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600, SERIAL_8N1, RADAR_RX, RADAR_TX);
  
  // Motor pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(EN_A, OUTPUT);
  pinMode(EN_B, OUTPUT);
  
  // Sensor pins
  pinMode(TRIG_LEFT, OUTPUT);
  pinMode(ECHO_LEFT, INPUT);
  pinMode(TRIG_RIGHT, OUTPUT);
  pinMode(ECHO_RIGHT, INPUT);
  pinMode(POT_PIN, INPUT);
  
  // PWM Setup (ESP32 Core v3 approach)
  // If this fails to compile on older cores, use ledcSetup/ledcAttachPin logic
  ledcAttachChannel(EN_B, PWM_FREQ, PWM_RES, DRIVE_CH);
  ledcAttachChannel(EN_A, PWM_FREQ, PWM_RES, STEER_CH);
  
  stopMotors();
  
  // WiFi AP Mode
  WiFi.softAP(ssid, password);
  Serial.println("AP Started");
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());
  
  // Web Server Routes
  server.on("/", handleRoot);
  server.on("/control", HTTP_POST, handleControl);
  server.on("/mode", HTTP_POST, handleMode);
  server.on("/stop", HTTP_POST, handleStop);
  server.on("/sensors", handleSensors);
  server.on("/python", HTTP_POST, handlePython);
  
  server.begin();
  Serial.println("Server started. Listening for Serial <ANGLE,SPEED>...");
}

void loop() {
  server.handleClient();
  readSensors();
  
  // Failsafe check
  if (millis() - lastCmdTime > TIMEOUT) {
    if (driveSpeed != 0 || steerAngle != 0) {
       stopMotors();
       driveSpeed = 0;
       steerAngle = 0;
    }
  }

  // Serial Parsing for Jetson
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '<') {
      inputString = "";
    } else if (inChar == '>') {
      stringComplete = true;
      break;
    } else {
      inputString += inChar;
    }
  }

  if (stringComplete) {
    parseSerialCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  delay(10); // Small delay
}

// ================= SERIAL COMMANDS =================
void parseSerialCommand(String data) {
  // Expected: "ANGLE,SPEED" (e.g. "90,100")
  int commaIndex = data.indexOf(',');
  if (commaIndex != -1) {
    int angleInput = data.substring(0, commaIndex).toInt(); // 0-180
    int speedInput = data.substring(commaIndex + 1).toInt(); // -255 to 255
    
    // Map Angle (0-180) to Steering PWM (-255 to 255)
    // 90 -> 0, 0 -> -255, 180 -> 255
    int steerPWM = map(angleInput, 0, 180, -255, 255);
    
    // Apply
    setDriveMotor(speedInput);
    setSteerMotor(steerPWM);
    
    driveSpeed = speedInput;
    steerAngle = steerPWM;
    lastCmdTime = millis();
    autoMode = true; // Force auto mode on serial command
  }
}

// ================= WEB HANDLERS =================
void handleRoot() {
  String html = R"(<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADAS Control</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Arial,sans-serif;background:#0a0e27;color:#fff;display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:20px}
.header{text-align:center;margin-bottom:30px}
.header h1{font-size:28px;margin-bottom:5px}
.status{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;justify-content:center}
.status-item{background:#1a1f3a;padding:10px 20px;border-radius:8px;font-size:14px}
.mode-toggle{background:#2a3f5f;padding:15px;border-radius:12px;margin-bottom:20px;text-align:center}
.toggle-btn{padding:12px 30px;border:none;border-radius:8px;font-size:16px;cursor:pointer;transition:all 0.3s}
.manual{background:#4a90e2;color:#fff}
.auto{background:#f39c12;color:#fff}
.controls{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin-bottom:20px}
.joystick-container{background:#1a1f3a;padding:20px;border-radius:12px;text-align:center}
.joystick{width:180px;height:180px;background:#0a0e27;border-radius:50%;position:relative;margin:10px auto;border:3px solid #2a3f5f;touch-action:none}
.stick{width:60px;height:60px;background:#4a90e2;border-radius:50%;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);cursor:pointer;box-shadow:0 0 20px rgba(74,144,226,0.5)}
.sensors{background:#1a1f3a;padding:20px;border-radius:12px;width:100%;max-width:600px;margin-bottom:20px}
.sensor-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:15px;margin-top:15px}
.sensor-box{background:#0a0e27;padding:15px;border-radius:8px;text-align:center}
.sensor-value{font-size:24px;color:#4a90e2;margin-top:5px}
.emergency{background:#e74c3c;color:#fff;padding:15px 40px;border:none;border-radius:8px;font-size:18px;cursor:pointer;margin:20px 0}
.emergency:active{background:#c0392b}
.camera{background:#1a1f3a;padding:20px;border-radius:12px;width:100%;max-width:640px}
.camera img{width:100%;border-radius:8px;background:#000}
</style>
</head>
<body>
<div class="header">
<h1>ADAS LEVEL 3</h1>
<div class="status">
<div class="status-item">Mode: <span id="mode">MANUAL</span></div>
<div class="status-item">Drive: <span id="drive">0</span></div>
<div class="status-item">Steer: <span id="steer">0</span></div>
</div>
</div>

<div class="mode-toggle">
<button id="modeBtn" class="toggle-btn manual" onclick="toggleMode()">MANUAL MODE</button>
</div>

<div class="controls">
<div class="joystick-container">
<h3>Drive Control</h3>
<div class="joystick" id="driveJoy"><div class="stick" id="driveStick"></div></div>
</div>
<div class="joystick-container">
<h3>Steering Control</h3>
<div class="joystick" id="steerJoy"><div class="stick" id="steerStick"></div></div>
</div>
</div>

<button class="emergency" onclick="emergencyStop()">EMERGENCY STOP</button>

<div class="sensors">
<h3>Sensor Data</h3>
<div class="sensor-grid">
<div class="sensor-box"><div>Left</div><div class="sensor-value" id="leftDist">0</div></div>
<div class="sensor-box"><div>Right</div><div class="sensor-value" id="rightDist">0</div></div>
<div class="sensor-box"><div>Radar</div><div class="sensor-value" id="radarDist">0</div></div>
<div class="sensor-box"><div>Position</div><div class="sensor-value" id="steerPos">0</div></div>
</div>
</div>

<script>
let auto=false;
function initJoystick(joyId,stickId,axis){
const joy=document.getElementById(joyId);
const stick=document.getElementById(stickId);
let active=false;
const move=(e)=>{
if(!active||auto)return;
const rect=joy.getBoundingClientRect();
const centerX=rect.width/2;
const centerY=rect.height/2;
let x=(e.clientX||e.touches[0].clientX)-rect.left-centerX;
let y=(e.clientY||e.touches[0].clientY)-rect.top-centerY;
const dist=Math.sqrt(x*x+y*y);
const maxDist=rect.width/2-30;
if(dist>maxDist){x=(x/dist)*maxDist;y=(y/dist)*maxDist;}
stick.style.transform='translate(calc(-50% + '+x+'px),calc(-50% + '+y+'px))';
let val=axis==='y'?-Math.round((y/maxDist)*255):Math.round((x/maxDist)*255);
sendControl(axis,val);
};
const end=()=>{
active=false;
stick.style.transform='translate(-50%,-50%)';
sendControl(axis,0);
};
joy.addEventListener('mousedown',()=>active=true);
joy.addEventListener('touchstart',()=>active=true);
document.addEventListener('mousemove',move);
document.addEventListener('touchmove',move);
document.addEventListener('mouseup',end);
document.addEventListener('touchend',end);
}
initJoystick('driveJoy','driveStick','y');
initJoystick('steerJoy','steerStick','x');

function sendControl(axis,val){
fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({axis:axis,value:val})});
}
function toggleMode(){
auto=!auto;
fetch('/mode',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({auto:auto})}).then(()=>{
document.getElementById('mode').textContent=auto?'AUTO':'MANUAL';
document.getElementById('modeBtn').textContent=auto?'AUTO MODE':'MANUAL MODE';
document.getElementById('modeBtn').className='toggle-btn '+(auto?'auto':'manual');
});
}
function emergencyStop(){
fetch('/stop',{method:'POST'});
}
setInterval(()=>{
fetch('/sensors').then(r=>r.json()).then(d=>{
document.getElementById('leftDist').textContent=d.left+'cm';
document.getElementById('rightDist').textContent=d.right+'cm';
document.getElementById('radarDist').textContent=d.radar+'cm';
document.getElementById('steerPos').textContent=d.pos+'°';
document.getElementById('drive').textContent=d.drive;
document.getElementById('steer').textContent=d.steer;
});
},200);
</script>
</body>
</html>)";
  server.send(200, "text/html", html);
}

void handleControl() {
  if (autoMode) {
    server.send(200, "text/plain", "Auto mode active");
    return;
  }
  
  StaticJsonDocument<200> doc;
  deserializeJson(doc, server.arg("plain"));
  
  String axis = doc["axis"];
  int value = doc["value"];
  
  if (axis == "y") {
    driveSpeed = value;
    setDriveMotor(value);
  } else if (axis == "x") {
    steerAngle = value;
    setSteerMotor(value);
  }
  
  lastCmdTime = millis();
  server.send(200, "text/plain", "OK");
}

void handleMode() {
  StaticJsonDocument<100> doc;
  deserializeJson(doc, server.arg("plain"));
  autoMode = doc["auto"];
  
  if (!autoMode) {
    stopMotors();
    driveSpeed = 0;
    steerAngle = 0;
  }
  
  Serial.print("Mode changed to: ");
  Serial.println(autoMode ? "AUTO" : "MANUAL");
  
  server.send(200, "text/plain", "OK");
}

void handleStop() {
  stopMotors();
  driveSpeed = 0;
  steerAngle = 0;
  server.send(200, "text/plain", "Stopped");
}

void handleSensors() {
  StaticJsonDocument<300> doc;
  doc["left"] = sensors.leftDist;
  doc["right"] = sensors.rightDist;
  doc["radar"] = sensors.radarDist;
  doc["pos"] = sensors.steerPos;
  doc["drive"] = driveSpeed;
  doc["steer"] = steerAngle;
  doc["auto"] = autoMode;
  
  String response;
  serializeJson(doc, response);
  server.send(200, "application/json", response);
}

void handlePython() {
  // Web Handler for Python (Optional backup if Serial fails)
  StaticJsonDocument<200> doc;
  deserializeJson(doc, server.arg("plain"));
  
  int drive = doc["drive"];
  int steer = doc["steer"];
  
  setDriveMotor(drive);
  setSteerMotor(steer);
  
  driveSpeed = drive;
  steerAngle = steer;
  lastCmdTime = millis();
  autoMode = true;
  
  server.send(200, "text/plain", "OK");
}

// ================= MOTOR CONTROL =================
void setDriveMotor(int speed) {
  speed = constrain(speed, -255, 255);
  
  if (speed > 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    ledcWrite(EN_B, speed);
  } else if (speed < 0) {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    ledcWrite(EN_B, -speed);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    ledcWrite(EN_B, 0);
  }
}

void setSteerMotor(int angle) {
  // Angle here is actually PWM intensity + Direction (-255 to 255)
  angle = constrain(angle, -255, 255);
  
  if (angle > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    ledcWrite(EN_A, abs(angle));
  } else if (angle < 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    ledcWrite(EN_A, abs(angle));
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    ledcWrite(EN_A, 0);
  }
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(EN_A, 0);
  ledcWrite(EN_B, 0);
}

// ================= SENSOR READING =================
void readSensors() {
  sensors.leftDist = getUltrasonicDistance(TRIG_LEFT, ECHO_LEFT);
  sensors.rightDist = getUltrasonicDistance(TRIG_RIGHT, ECHO_RIGHT);
  sensors.radarDist = getRadarDistance();
  sensors.steerPos = map(analogRead(POT_PIN), 0, 4095, -90, 90);
}

float getUltrasonicDistance(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  
  long duration = pulseIn(echo, HIGH, 30000);
  if (duration == 0) return 999;
  return duration * 0.034 / 2;
}

float getRadarDistance() {
  if (Serial2.available()) {
    String data = Serial2.readStringUntil('\n');
    float dist = data.toFloat();
    if (dist > 0) return dist;
  }
  return sensors.radarDist;
}
