git pull

docker stop $(docker ps -a | grep "amazon-rec" | awk '{print $1 }')
docker rm $(docker ps -a | grep "amazon-rec" | awk '{print $1 }')

docker build . -t amazon-rec

docker run --name amazon-rec -d -p 80:80 -p 443:443  amazon-rec