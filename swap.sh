#!/bin/bash

# Größe der Swap-Datei in GB
swap_size=8

# Swap-Datei erstellen
sudo fallocate -l ${swap_size}G /swapfile

# Zugriffsrechte der Swap-Datei ändern
sudo chmod 600 /swapfile

# Swap-Datei als Swap-Partition formatieren
sudo mkswap /swapfile

# Swap-Partition aktivieren
sudo swapon /swapfile

# Swap-Partition dauerhaft aktivieren
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Ausgabe der Swap-Partition überprüfen
free -h
