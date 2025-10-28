#!/bin/bash
echo "Liberando memoria cache..."
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "✅ Memoria liberada"
free -h