# Hướng dẫn triển khai — Drowsiness Detection Controller

> **Mục tiêu:** biến project main.py thành sản phẩm đứng độc lập trên Pi 4:
> cắm điện là LED xanh nháy, bấm button khởi động/tắt, có LED đỏ + buzzer cảnh báo trực tiếp.
>
> **Môi trường:** Raspberry Pi 4 · Debian 13 (Trixie) · Python 3.13 hệ thống · venv 3.11 cho project (đã có sẵn theo `INSTALL_GUIDE_PITFALLS.md`).

---

## 1. Sơ đồ phần cứng

```
                       Raspberry Pi 4 (40-pin header)
                       ───────────────────────────────
   LED xanh ────[330Ω]──┤ GPIO22 (pin 15)
        anode           │
   cathode ──────────────┤ GND     (pin 14 hoặc 6, 9, 20...)

   LED đỏ ────[330Ω]──┤ GPIO17 (pin 11)
        anode           │
   cathode ──────────────┤ GND     (dùng GND chung)

   Active buzzer (+) ────┤ GPIO27 (pin 13)
   Active buzzer (−) ────┤ GND
   (* Active buzzer kéo dòng < 25mA — có thể nối thẳng. Nếu module
      của bạn ghi > 25mA hoặc 5V thì PHẢI qua transistor + 5V.)

   Button: 2 chân  ──────┤ GPIO26 (pin 37)
                         │
                  ────────┤ GND     (pin 39, hoặc GND khác)
   (Không cần điện trở ngoài — code bật pull-up nội. Bấm = kéo xuống LOW.)
```

**Điện trở cho LED:** 220Ω–470Ω (330Ω là sweet spot cho LED 2V/20mA với nguồn 3.3V Pi).
Không bỏ điện trở — sẽ cháy LED và có thể hỏng GPIO của Pi.

**Pin layout tham chiếu:** `RP008341DS1raspberrypi4datasheet.pdf` mục GPIO Pin Assignments.

---

## 2. Cấu trúc file

Sau khi triển khai, project của bạn sẽ có thêm 3 file:

```
~/Documents/ce_comedians/project/
├── main.py                  ← chỉnh sửa 3 vị trí (xem mục 4)
├── main_gpio_alert.py       ← MỚI — driver LED đỏ + buzzer
├── pi_controller.py         ← MỚI — boot controller (LED xanh + button)
├── pi-controller.service    ← MỚI — systemd user unit
└── venv/                    ← venv 3.11 đã có
```

---

## 3. Cài đặt thư viện GPIO

### 3.1. Cho controller (Python 3.13 hệ thống)

Trên Raspberry Pi OS Trixie, `python3-gpiozero` và `python3-lgpio` thường đã có sẵn,
nhưng cứ chạy để chắc chắn:

```bash
sudo apt update
sudo apt install -y python3-gpiozero python3-lgpio
```

Kiểm tra:
```bash
/usr/bin/python3 -c "from gpiozero import LED; from gpiozero.pins.lgpio import LGPIOFactory; print('OK')"
```

### 3.2. Cho main.py (venv 3.11)

Đây là phần **dễ vấp** vì lgpio chưa có wheel sẵn cho Python 3.11 trên ARM64,
phải compile từ source theo gói C `liblgpio-dev`.

```bash
# 1) Cài thư viện C system-wide
sudo apt install -y liblgpio-dev

# 2) Activate venv 3.11 và cài
cd ~/Documents/ce_comedians/project
source venv/bin/activate
pip install --upgrade pip
pip install gpiozero lgpio

# 3) Test trong venv (LƯU Ý: dùng `python3`, KHÔNG dùng `/usr/bin/python3`
#    vì /usr/bin/python3 sẽ trỏ vào Python 3.13 hệ thống, bypass venv)
python3 -c "from gpiozero import LED, Buzzer; from gpiozero.pins.lgpio import LGPIOFactory; print('venv GPIO OK')"
```

**Nếu `pip install lgpio` báo lỗi `cannot find -llgpio`:**
chứng tỏ `liblgpio-dev` chưa cài hoặc thiếu. Cài lại bước 1, hoặc compile thủ công
theo `https://abyz.me.uk/lg/download.html` (chỉ cần nếu apt thiếu).

---

## 4. Sửa `main.py` — chèn 3 vị trí

## 5. Đặt 3 file mới vào project

```bash
# Giả định bạn đã có 3 file: main_gpio_alert.py, pi_controller.py, pi-controller.service
cd ~/Documents/ce_comedians/project
chmod +x pi_controller.py

# Test thử main_gpio_alert độc lập (KHÔNG cần camera)
source venv/bin/activate
python3 -c "
import main_gpio_alert, time
print('Bat alert 4s...')
for _ in range(80):    # 80 * 0.05 = 4s
    main_gpio_alert.update(True)
    time.sleep(0.05)
print('Tat alert 1s...')
for _ in range(20):
    main_gpio_alert.update(False)
    time.sleep(0.05)
main_gpio_alert.cleanup()
"
```

**Kết quả mong đợi:**
- 0–2s: LED đỏ nhấp nháy nhanh (chu kỳ 0.5s), buzzer im.
- 2–4s: LED đỏ + buzzer cùng nháy chậm hơn (chu kỳ 1s).
- Sau đó: tắt hoàn toàn.

Nếu LED đỏ không sáng → kiểm tra chiều LED (anode/cathode), điện trở, GND chung.
Nếu buzzer chỉ "tick" mà không kêu liên tục → nó là **passive buzzer**, không phải active
(lúc đó cần đổi sang `Buzzer.beep()` với PWM — báo lại cho tôi).

---

## 6. Test controller thủ công (trước khi đăng ký systemd)

```bash
deactivate   # rời venv 3.11
cd ~/Documents/ce_comedians/project
/usr/bin/python3 pi_controller.py
```

**Quan sát:**
1. LED xanh bắt đầu nháy mỗi 1s.
2. Bấm button → cửa sổ camera mở (qua main.py), LED xanh sáng liên tục.
3. Bấm button lần 2 → trong vòng ≤ 8s, cửa sổ camera đóng, LED đỏ + buzzer tắt sạch,
   LED xanh quay về nháy 1Hz.
4. Lặp lại tùy ý.

Dừng controller: `Ctrl+C` — controller sẽ tự gửi SIGTERM cho main.py rồi tắt LED.

**Nếu bấm button không phản ứng:** kiểm tra `gpio readall` (cài `sudo apt install raspi-gpio`)
xem pin 37 có ở chế độ INPUT/pull-up không. Hoặc test trực tiếp:
```bash
python3 -c "
from gpiozero import Button
b = Button(26, pull_up=True)
b.wait_for_press()
print('Da bam!')
"
```

---

## 7. Đăng ký systemd user service — chạy tự động khi boot

User đã bật **graphical autologin**, nên user systemd sẽ tự khởi động khi vào desktop.

```bash
# 1) Copy unit vao thu muc user systemd
mkdir -p ~/.config/systemd/user
cp ~/Documents/ce_comedians/project/pi-controller.service \
   ~/.config/systemd/user/pi-controller.service

# 2) Reload + enable + start
systemctl --user daemon-reload
systemctl --user enable pi-controller.service
systemctl --user start  pi-controller.service

# 3) Kiem tra
systemctl --user status pi-controller.service
journalctl --user -u pi-controller.service -f   # theo doi log realtime
```

**Test reboot:**
```bash
sudo reboot
# Khi Pi vao desktop, LED xanh phai bat dau nhay 1Hz trong vong vai giay.
```

### Vì sao dùng USER service mà không phải SYSTEM service?

| | User service | System service |
|---|---|---|
| DISPLAY=:0 cho cv2.imshow | ✓ tự động (inherit từ session) | Phải set `Environment=DISPLAY=:0` + `XAUTHORITY=...` |
| Phụ thuộc graphical session | ✓ tự nhiên (PartOf=graphical-session) | Phải `After=graphical.target` + tricks |
| Quyền GPIO | ✓ user `phat` ở group `gpio` | ✓ root, hoặc user nếu set User= |
| Dừng khi user logout | ✓ (đúng hành vi mong muốn) | ✗ tiếp tục chạy |
| Debug | `journalctl --user -u ...` | `sudo journalctl -u ...` |

Với autologin graphical, user service là lựa chọn sạch hơn rất nhiều.

---

## 8. Pattern cảnh báo — minh họa thời gian

```
Trạng thái alert:    OFF │ ON ────────────────────────────────────── │ OFF
Thời gian (s):       ────┼─0────0.5───1───1.5───2────2.5───3────3.5──┼────
                         │                                            │
LED đỏ:             OFF │ ░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓ │ ▓▓▓▓▓░░░░░▓▓▓▓▓░░░░░ │ OFF
                         │ ──── Phase 1 (0–2s) ──── │ ── Phase 2 (>2s) ── │
                         │ 0.25 on / 0.25 off       │ 0.5 on / 0.5 off     │
                         │ buzzer IM                │ buzzer KÊU đồng bộ   │
Buzzer:              OFF │ ──────────────────────── │ ▓▓▓▓▓░░░░░▓▓▓▓▓░░░░░ │ OFF
```

Khi `alert_messages` đột ngột rỗng (ví dụ người dùng tỉnh lại, mặt trở lại bình thường):
→ LED đỏ + buzzer tắt **ngay frame kế tiếp**, đồng hồ phase reset về 0.

---

## 9. Gỡ rối phổ biến

| Triệu chứng | Nguyên nhân khả năng cao | Cách kiểm tra |
|---|---|---|
| LED xanh không nháy sau khi service start | Service chưa active hoặc lỗi GPIO | `systemctl --user status pi-controller` + `journalctl --user -u pi-controller -n 50` |
| Bấm button không có gì | Sai pin, button hỏng, pull-up không hoạt động | Test isolated với `Button(26, pull_up=True)` |
| `main.py` không mở cửa sổ camera | DISPLAY chưa set | Trong unit: thêm `Environment=DISPLAY=:0` (thường không cần cho user service) |
| LED đỏ/buzzer không phản ứng khi có alert | Module `main_gpio_alert` import lỗi | Trong main.py log có `[GPIO_ALERT] Khong khoi tao duoc...` |
| Bấm button không tắt được main.py (treo) | `cap.release()` block | Controller sẽ SIGKILL sau 8s; tăng `SIGTERM_TIMEOUT` nếu cần |
| GPIO bị giữ sau crash | Process trước chưa release | `sudo lgpio-info` hoặc reboot |
| Service không tự start sau reboot | Chưa `enable`, hoặc lingering chưa bật | `systemctl --user is-enabled pi-controller`; nếu cần: `sudo loginctl enable-linger phat` |

---

## 10. Tham chiếu

- Raspberry Pi (12/2025) — Trixie release notes: https://www.raspberrypi.com/news/trixie-the-new-version-of-raspberry-pi-os/
  Xác nhận Python 3.13 default, `python3-lgpio` cài system-wide.
- gpiozero documentation: https://gpiozero.readthedocs.io/
- gpiozero pin factories (lgpio): https://gpiozero.readthedocs.io/en/latest/api_pins.html
- lgpio C library: https://abyz.me.uk/lg/lgpio.html
- systemd user services: https://www.freedesktop.org/software/systemd/man/systemd.unit.html
- Pi 4 GPIO map: `RP008341DS1raspberrypi4datasheet.pdf` (file đính kèm project)
