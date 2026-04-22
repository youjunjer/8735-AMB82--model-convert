# 8735(AMB82) Teachable Machine 模型轉換服務

這個專案提供一個可部署的 Web UI，讓使用者上傳 `converted_keras.zip` 與校正圖片，背景呼叫 Acuity Docker 轉換流程，最後產出 `imgclassification.nb`，並透過 email 通知下載連結。

正式站：
- [https://model.mqttgo.io](https://model.mqttgo.io)

GitHub：
- [https://github.com/youjunjer/8735-AMB82--model-convert](https://github.com/youjunjer/8735-AMB82--model-convert)

## 這份文件的目標

這份 README 不是只有開發說明，而是要讓「另一位 AI 或工程師只看 repo」就能從零重建整個服務。

如果你是新的接手者，建議照這個順序看：

1. 先看「整體架構」
2. 再做「從零重建步驟」
3. 最後看「故障排查清單」

## 目前功能

- 上傳 `converted_keras.zip`
- 至少 1 張校正圖片
- 單一 worker 排隊轉換
- 顯示目前轉換中的工作編號
- 顯示目前排隊中的工作編號
- 累計完成次數持久保存
- Gmail SMTP 三段通知
  - 已收到工作
  - 轉換成功
  - 轉換失敗
- `model.mqttgo.io` HTTPS 對外服務

## 整體架構

### 1. 前台

- Python FastAPI Web UI
- 檔案上傳
- 驗證碼
- 工作排隊
- 狀態顯示
- 成功/失敗/收到工作通知信

### 2. 背景轉換

- 使用 WSL 發行版：`AMB_Model`
- 在 WSL 內執行 Docker Engine
- 呼叫 `ghcr.io/ameba-aiot/acuity-toolkit:6.18.8`

### 3. 模型轉換流程

1. 解壓 `converted_keras.zip`
2. 取出 `keras_model.h5`
3. `h5 -> ONNX`
4. `ONNX -> Acuity import`
5. `quantize`
6. `export -> imgclassification.nb`

### 4. 對外服務

- WSL 內的 Web UI 綁定 `0.0.0.0:8891`
- Caddy 在 WSL 內提供 `80/443`
- Windows 主機用 `portproxy` 轉發 `80/443 -> WSL`
- 公網 DNS 指向 Windows 主機外網 IP

### 5. 持久資料

- `webui_jobs/`
  - 每筆工作的輸入、log、輸出
- `service_state.json`
  - 累計完成次數

## 專案結構

- `model_convert_webui.py`
  Web UI、API、排隊、mail 通知主程式
- `zip_to_nb_wsl.sh`
  `converted_keras.zip -> imgclassification.nb` 的主轉換流程
- `zip_to_nb_wsl_gpu.sh`
  GPU 包裝版轉換腳本
- `run_model_convert_webui_wsl.sh`
  WSL / systemd 啟動腳本
- `deployment/amb82-model-convert.service`
  systemd 服務範本
- `deployment/Caddyfile`
  Caddy 反向代理範本
- `service_icons/`
  網站與 mail 圖示
- `site_assets/`
  favicon 與站台資產
- `webui_jobs/`
  執行期工作目錄
- `service_state.json`
  累計完成次數統計

## 必要環境

- Windows 11
- WSL2
- 一個專用 WSL 發行版
  - 本次名稱：`AMB_Model`
- WSL 內可正常使用 Docker Engine
- Docker 可拉取：
  - `ghcr.io/ameba-aiot/acuity-toolkit:6.18.8`
- Python 3.10+
- 可用的 Gmail SMTP App Password
- 一個可以設定 DNS 的網域

## 從零重建步驟

### 1. 建立或匯入 WSL

範例：

```powershell
wsl --import AMB_Model E:\wsl\AMB_Model <tar-or-rootfs>
```

建議先確認：

```powershell
wsl -l -v
wsl -d AMB_Model -- uname -a
```

### 2. 安裝 WSL 內的基本套件

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl git ca-certificates gnupg lsb-release caddy
```

### 3. 安裝 WSL 內的 Docker Engine

這份 repo 假設你是「在 WSL 內自己跑 Docker Engine」，不是依賴 Docker Desktop integration。

官方方式可參考 Docker Engine for Ubuntu，實際最小步驟如下：

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

重新登入 WSL 後確認：

```bash
docker version
docker ps
docker images
```

如果 `docker ps` 失敗，先不要往下做。

### 4. 申請並驗證 Acuity / GHCR 權限

這一步是最容易被忽略的。沒有權限時，後面所有轉換都會卡住。

你需要：

1. GitHub 帳號取得 Realtek / Ameba 離線工具權限
2. 可存取對應的私有 package / repo
3. 有一組 GitHub PAT

官方文件：

- [Acuity Toolkit Docker Installation](https://ameba-doc-ai-video-analytics-doc.readthedocs-hosted.com/en/latest/user_manual/Acuity_tool/Acuity_installation_docker.html)

PAT 最少建議包含：

- `read:packages`
- `write:packages`
- `delete:packages`

登入：

```bash
docker login ghcr.io -u <github_username>
```

驗證：

```bash
docker pull ghcr.io/ameba-aiot/acuity-toolkit:6.18.8
```

如果你看到：

- `401 unauthorized`
  - 通常是沒登入或 token 錯
- `403 forbidden`
  - 通常是 GHCR / repo 權限未開通

### 5. 下載專案

```bash
git clone https://github.com/youjunjer/8735-AMB82--model-convert.git
cd 8735-AMB82--model-convert
```

### 6. 建立 `.env`

```bash
cp .env.example .env
```

最少要填：

```env
MODEL_WSL_DISTRO=AMB_Model
MODEL_WEBUI_HOST=0.0.0.0
MODEL_WEBUI_PORT=8891
MODEL_WEBUI_BASE_URL=https://model.mqttgo.io

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=<your gmail>
SMTP_PASSWORD=<gmail app password>
SMTP_FROM_EMAIL=<your gmail>
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

### 7. 先手動啟動一次

WSL 內：

```bash
bash ./run_model_convert_webui_wsl.sh
```

這支腳本會：

1. 建立 Python venv
2. 安裝 `requirements.txt`
3. 啟動 `model_convert_webui.py`

健康檢查：

```bash
curl http://127.0.0.1:8891/health
```

預期：

```json
{"status":"ok","wsl_distro":"AMB_Model"}
```

### 8. 註冊成 systemd 服務

```bash
sudo cp deployment/amb82-model-convert.service /etc/systemd/system/amb82-model-convert.service
sudo systemctl daemon-reload
sudo systemctl enable amb82-model-convert.service
sudo systemctl restart amb82-model-convert.service
```

檢查：

```bash
systemctl status amb82-model-convert.service
journalctl -u amb82-model-convert.service -n 100 --no-pager
```

### 9. 設定 Caddy HTTPS

先編輯 `deployment/Caddyfile`：

- 把裡面的 email 改成你自己的憑證通知信箱
- 如果網域不是 `model.mqttgo.io`，就改成你的正式網域

然後套用：

```bash
sudo cp deployment/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

這份設定的預設是：

```text
https://model.mqttgo.io -> http://127.0.0.1:8891
```

### 10. Windows 設定 80 / 443 轉發

這一步需要「系統管理員 PowerShell」。

先查 WSL IP：

```powershell
wsl -d AMB_Model -- hostname -I
```

假設回傳是 `172.28.155.27`，則新增：

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=80 connectaddress=172.28.155.27 connectport=80
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=443 connectaddress=172.28.155.27 connectport=443
netsh advfirewall firewall add rule name="model.mqttgo.io HTTP" dir=in action=allow protocol=TCP localport=80
netsh advfirewall firewall add rule name="model.mqttgo.io HTTPS" dir=in action=allow protocol=TCP localport=443
```

查看：

```powershell
netsh interface portproxy show all
```

刪除：

```powershell
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=80
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=443
```

如果 WSL IP 變了，請先刪掉舊規則，再重新建立新規則。

### 11. DNS

把：

- `model.mqttgo.io`

指向這台 Windows 主機的外網 IP。

### 12. 完整驗證

本機驗證：

```bash
curl http://127.0.0.1:8891/health
```

外網驗證：

```powershell
curl https://model.mqttgo.io/health
```

再實際上傳一筆 `converted_keras.zip` + 校正圖片，驗證：

1. 前台可建立工作
2. 會先收到「已收到工作」通知
3. 轉換完成後會收到成功信
4. 下載連結可下載 `imgclassification.nb`

## Mail 通知邏輯

目前共有三類信：

1. 已收到工作
2. 轉換成功
3. 轉換失敗

寄件者名稱固定為：

- `NMKING小霸王實驗室`

如果要重測 SMTP，可在 WSL 內直接匯入：

- `send_received_email`
- `send_job_email`

## 持久資料

以下資料不應該隨便刪：

- `webui_jobs/`
  - 已完成工作的 log 與輸出
- `service_state.json`
  - 累計完成次數

如果只刪掉程式而沒有備份這兩者：

- 已完成工作記錄會消失
- 累計完成次數會重建或歸零

## 後續 AI 接手時的最小檢查清單

### 快速確認

1. `https://model.mqttgo.io/health` 是否正常
2. `systemctl status amb82-model-convert.service`
3. `docker ps`
4. `docker images | grep acuity-toolkit`
5. `.env` 是否存在且 SMTP 正確
6. `systemctl status caddy`
7. `netsh interface portproxy show all`

### 常見故障

- WSL IP 變動
  - `80/443` 轉發失效
- Gmail App Password 失效
  - 收不到信
- Acuity GHCR 權限過期
  - `docker pull` 失敗
- Docker 在 WSL 內沒起來
  - 所有轉換卡住
- `.env` 被覆蓋
  - BASE URL / SMTP / Host 錯誤
- Caddyfile 沒改 email 或網域
  - 憑證申請失敗

## GitHub 同步

目前 repo 遠端應為：

```text
origin https://github.com/youjunjer/8735-AMB82--model-convert.git
```

日後同步：

```bash
git add .
git commit -m "your message"
git push
```

## 模擬結論

這份 README 的目標，是讓新的 AI 在沒有對話歷史的情況下，也能只靠 repo 重建服務。

如果照這份文件做，理論上可以重建：

- WSL 服務
- Docker / Acuity 轉換流程
- HTTPS 對外入口
- SMTP 通知
- Windows portproxy

剩下最依賴外部條件的只有：

- 你是否已取得 GHCR / Realtek 權限
- 你的 DNS 是否已正確指向主機
- 你的 Gmail App Password 是否可用
