# needed to download model from HF Hub
sysctl net.ipv6.conf.all.disable_ipv6=1
# allows HTTP and HTPS ansd SSH
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw reload
