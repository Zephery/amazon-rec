git pull

docker stop $(docker ps -a | grep "amazon-rec" | awk '{print $1 }')
docker rm $(docker ps -a | grep "amazon-rec" | awk '{print $1 }')

docker build . -t amazon-rec

docker run --name amazon-rec -d -p 80:80 -p 443:443  amazon-rec


docker stop $(docker ps -a | grep "rec-flask" | awk '{print $1 }')
docker rm $(docker ps -a | grep "rec-flask" | awk '{print $1 }')

cd rec-flask && docker build . -t rec-flask

docker run --name rec-flask -d -p 5000:5000  rec-flask
