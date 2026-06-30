import serial
import threading
import time
import queue

class SerialController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.command_queue = queue.Queue(maxsize=1)
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True
        self.thread.start()
        return self

    def send_command(self, angle, speed):
        """
        Send control command to ESP32.
        angle: 0-180 (90 center)
        speed: -255 to 255
        """
        if not self.running: return
        
        # Format: <ANGLE,SPEED>
        cmd = f"<{int(angle)},{int(speed)}>"
        
        # Update latest command (overwrite old if queue full)
        if self.command_queue.full():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass
        self.command_queue.put(cmd)

    def _connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Serial connected: {self.port}")
            self._reconnect_delay = 2  # Reset backoff on success
            time.sleep(2) # Wait for ESP32 reset
            return True
        except serial.SerialException as e:
            if not hasattr(self, '_last_error') or str(e) != self._last_error:
                print(f"Serial connection failed: {e}")
                self._last_error = str(e)
            return False

    def _run_loop(self):
        self._reconnect_delay = 2
        while self.running:
            if self.serial is None or not self.serial.is_open:
                if not self._connect():
                    time.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 2, 30)  # Backoff up to 30s
                    continue

            try:
                # 1. Send Command
                try:
                    cmd = self.command_queue.get(timeout=0.1)
                    if self.serial and self.serial.is_open:
                        self.serial.write(cmd.encode())
                        # print(f"Sent: {cmd}") # Debug
                except queue.Empty:
                    pass

                # 2. Read Response (optional logging)
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"ESP32: {line}")
            
            except Exception as e:
                print(f"Serial Loop Error: {e}")
                if self.serial:
                    self.serial.close()
                time.sleep(1)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.serial and self.serial.is_open:
            self.serial.close()
