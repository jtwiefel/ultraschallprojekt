sudo insmod ./scope.ko
sudo mknod /dev/chardev c 243 0
echo modules loaded:
lsmod | grep scope
ls /dev/ | grep char
cat /dev/chardev > $1
sudo rm /dev/chardev
sudo rmmod scope.ko
echo modules left:
lsmod | grep scope
ls /dev/ | grep char
