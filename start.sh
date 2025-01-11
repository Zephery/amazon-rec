git pull

docker stop amazon-rec

docker build . -t amazon-rec

docker run --name amazon-rec -d -p 80:80 amazon-rec