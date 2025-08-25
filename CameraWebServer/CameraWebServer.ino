#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// --- LED Pin (Flash LED for AI Thinker is GPIO 4) ---
#define LED_PIN 4  

// WiFi AP credentials
const char *ap_ssid     = "esp_cam-blind";
const char *ap_password = "12345678";

WebServer server(80);

// Static IP for AP
IPAddress local_IP(192,168,4,1);
IPAddress gateway(192,168,4,1);
IPAddress subnet(255,255,255,0);

void setupLedServer() {
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/plain", "ESP32-CAM is running.\nTry /led?on or /led?off\nStream: http://192.168.4.1:81/stream");
  });

  server.on("/led", HTTP_GET, []() {
  if (server.hasArg("state")) {
    String val = server.arg("state");
    if (val == "on") {
      digitalWrite(LED_PIN, HIGH);
      server.send(200, "text/plain", "LED ON");
    } else if (val == "off") {
      digitalWrite(LED_PIN, LOW);
      server.send(200, "text/plain", "LED OFF");
    } else {
      server.send(400, "text/plain", "Invalid value. Use ?state=on or ?state=off");
    }
  } else {
    server.send(400, "text/plain", "Missing arg. Use ?state=on or ?state=off");
  }
});


  server.begin();
}

void startCameraServer();  // from app_httpd.cpp (keeps :81 stream)

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // --- Camera config (same as example) ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Init camera
  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed!");
    return;
  }

  // LED pin setup
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // WiFi AP mode with static IP
  if (!WiFi.softAPConfig(local_IP, gateway, subnet)) {
    Serial.println("AP Config Failed!");
  }
  WiFi.softAP(ap_ssid, ap_password);

  Serial.print("WiFi AP started. IP: ");
  Serial.println(WiFi.softAPIP());

  // Start both servers
  setupLedServer();   // port 80
  startCameraServer(); // port 81
}

void loop() {
  server.handleClient();  // handle LED requests
}
