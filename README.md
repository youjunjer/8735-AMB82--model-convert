# 8735(AMB82) Teachable Machine 模型轉換服務

這個專案提供一個可部署的 Web UI，讓使用者上傳 `converted_keras.zip` 與校正圖片，背景呼叫 Acuity Docker 轉換流程，最後產出 `network_binary.nb`，並透過 email 通知下載連結。

正式站：
- [https://model.mqttgo.io](https://model.mqttgo.io)

GitHub：
- [https://github.com/youjunjer/8735-AMB82--model-convert](https://github.com/youjunjer/8735-AMB82--model-convert)

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

## 專案結構

- [model_convert_webui.py](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/model_convert_webui.py)
  Web UI、API、排隊、mail 通知主程式
- [zip_to_nb_wsl.sh](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/zip_to_nb_wsl.sh)
  `converted_keras.zip -> network_binary.nb` 的主轉換流程
- [zip_to_nb_wsl_gpu.sh](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/zip_to_nb_wsl_gpu.sh)
  GPU 包裝版轉換腳本
- [run_model_convert_webui_wsl.sh](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/run_model_convert_webui_wsl.sh)
  WSL / systemd 啟動腳本
- [deployment/amb82-model-convert.service](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/deployment/amb82-model-convert.service)
  systemd 服務範本
- [deployment/Caddyfile](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/deployment/Caddyfile)
  Caddy 反向代理範本
- [service_icons](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/service_icons)
  網站與 mail 圖示
- [site_assets](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/site_assets)
  favicon 與站台資產
- [webui_jobs](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/webui_jobs)
  執行期工作目錄
- [service_state.json](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/service_state.json)
  累計完成次數統計

## 必要環境

- Windows 11
- WSL2
- 一個專用 WSL 發行版
  - 這次使用名稱：`AMB_Model`
- WSL 內可正常使用 Docker Engine
- Docker 可拉取：
  - `ghcr.io/ameba-aiot/acuity-toolkit:6.18.8`
- Python 3.10+
- 可用的 Gmail SMTP App Password
- 網域 DNS 可指向本機對外 IP

## 轉換流程

網站表面流程：

1. 使用者先到 Google Teachable Machine 建立模型
2. 匯出 `converted_keras.zip`
3. 再上傳至少 1 張校正圖片
4. 網站建立工作並排隊
5. 背景轉換完成後寄送下載連結

後端實際流程：

1. 解壓 `converted_keras.zip`
2. 取出 `keras_model.h5`
3. `h5 -> ONNX`
4. `ONNX -> Acuity import`
5. `quantize`
6. `export -> network_binary.nb`

## 重新建立本服務

### 1. 準備 WSL

建立或匯入一個 WSL 發行版，例如：

```powershell
wsl --import AMB_Model E:\wsl\AMB_Model <tar-or-rootfs>
```


建議安裝：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl git caddy
```

### 2. 讓 WSL 內可使用 Docker

這個專案目前是依賴「WSL 內可用的 Docker」。實作可以二選一：

- 直接使用 WSL 內自己的 Docker Engine
- 或先接 Docker Desktop integration

本專案最後採用的是：
- `AMB_Model` 內自己的 Docker Engine

確認指令：

```bash
docker ps
docker images
```

### 3. 確認可拉 Acuity image

```bash
docker pull ghcr.io/ameba-aiot/acuity-toolkit:6.18.8
```

如果是私有權限問題，要先：

- GitHub 帳號取得 `ameba-ai-offline-toolkit` 權限
- `docker login ghcr.io`

### 4. 下載專案

```bash
git clone https://github.com/youjunjer/8735-AMB82--model-convert.git
cd 8735-AMB82--model-convert
```

### 5. 建立 `.env`

複製範本：

```bash
cp .env.example .env
```

最少要填：

- `MODEL_WSL_DISTRO=AMB_Model`
- `MODEL_WEBUI_HOST=0.0.0.0`
- `MODEL_WEBUI_PORT=8891`
- `MODEL_WEBUI_BASE_URL=https://model.mqttgo.io`
- `SMTP_HOST=smtp.gmail.com`
- `SMTP_PORT=587`
- `SMTP_USERNAME=<your gmail>`
- `SMTP_PASSWORD=<gmail app password>`
- `SMTP_FROM_EMAIL=<your gmail>`
- `SMTP_USE_TLS=true`
- `SMTP_USE_SSL=false`

### 6. 本機先手動啟動一次

WSL 內：

```bash
bash ./run_model_convert_webui_wsl.sh
```

或 Windows PowerShell：

```powershell
.\run_model_convert_webui.ps1
```

測試健康檢查：

```bash
curl http://127.0.0.1:8891/health
```

### 7. 註冊成 systemd 服務

把 [deployment/amb82-model-convert.service](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/deployment/amb82-model-convert.service) 複製到：

```bash
sudo cp deployment/amb82-model-convert.service /etc/systemd/system/amb82-model-convert.service
sudo systemctl daemon-reload
sudo systemctl enable amb82-model-convert.service
sudo systemctl restart amb82-model-convert.service
```

常用指令：

```bash
systemctl status amb82-model-convert.service
sudo systemctl restart amb82-model-convert.service
journalctl -u amb82-model-convert.service -n 100 --no-pager
```

### 8. 設定 Caddy HTTPS

把 [deployment/Caddyfile](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/deployment/Caddyfile) 複製到：

```bash
sudo cp deployment/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

現在的代理方式是：

```text
https://model.mqttgo.io -> http://127.0.0.1:8891
```

### 9. Windows 對外轉發 80 / 443

這一步需要「系統管理員 PowerShell」。

先查 WSL IP：

```powershell
wsl -d AMB_Model -- hostname -I
```

再建立轉發：

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=80 connectaddress=<WSL_IP> connectport=80
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=443 connectaddress=<WSL_IP> connectport=443
netsh advfirewall firewall add rule name="model.mqttgo.io HTTP" dir=in action=allow protocol=TCP localport=80
netsh advfirewall firewall add rule name="model.mqttgo.io HTTPS" dir=in action=allow protocol=TCP localport=443
```

確認：

```powershell
netsh interface portproxy show all
```

注意：

- WSL IP 改變時，這裡要跟著重設
- 如果未來要長期維護，建議再做自動更新 portproxy 的腳本

### 10. DNS

把：

- `model.mqttgo.io`

指向這台 Windows 主機的對外 IP。

### 11. 驗證

```powershell
curl https://model.mqttgo.io/health
```

應回：

```json
{"status":"ok","wsl_distro":"AMB_Model"}
```

## Mail 通知邏輯

目前共有三類信：

1. 已收到工作
2. 轉換成功
3. 轉換失敗

寄件者名稱固定為：

- `NMKING小霸王實驗室`

如果要重測 SMTP，可以直接在 WSL 內匯入：

- `send_received_email`
- `send_job_email`

## 持久資料

以下資料不應該隨便刪：

- [webui_jobs](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/webui_jobs)
  - 已完成工作的 log 與輸出
- [service_state.json](/F:/GoogleDrv/P_程式開發/202604_AMB模型轉換/service_state.json)
  - 累計完成次數

如果只刪掉程式而沒有備份這兩者：

- 已完成工作記錄會消失
- 累計完成次數會重建或歸零

## GitHub 同步

目前 repo 遠端：

```bash
git remote -v
```

應為：

```text
origin https://github.com/youjunjer/8735-AMB82--model-convert.git
```

日後同步：

```bash
git add .
git commit -m "your message"
git push
```

## 封存備註

這個專案目前已可運作，適合封存。後續 AI 若要接手，優先順序建議如下：

1. 先確認 [https://model.mqttgo.io/health](https://model.mqttgo.io/health)
2. 再確認 WSL 服務 `amb82-model-convert.service`
3. 再確認 `docker ps` 與 Acuity image
4. 再檢查 `.env`
5. 最後才檢查 Caddy / DNS / Windows portproxy

最容易出問題的點：

- WSL IP 變動，導致 `80/443` 轉發失效
- Gmail App Password 失效
- Acuity GHCR 權限過期
- Docker 在 WSL 內沒起來
- `.env` 被覆蓋

