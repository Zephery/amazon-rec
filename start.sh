git pull

docker stop --name amazon-rec

docker build . -t amazon-rec

docker run --name amazon-rec -d -p 80:80 -p 443:443  amazon-rec