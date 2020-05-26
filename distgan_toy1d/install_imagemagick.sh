sudo apt update 
sudo apt-get install build-essential
wget https://www.imagemagick.org/download/ImageMagick.tar.gz
tar xvzf ImageMagick.tar.gz
rm ImageMagick.tar.gz
cd ImageMagick-*
./configure
make
sudo make install 
sudo ldconfig /usr/local/lib