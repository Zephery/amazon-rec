git pull

docker stop $(docker ps -a | grep "amazon-rec" | awk '{print $1 }')
docker rm $(docker ps -a | grep "amazon-rec" | awk '{print $1 }')

docker build . -t amazon-rec

docker run --name amazon-rec -d -p 80:80 -p 443:443  amazon-rec



(
    cd /home/admin/frp_0.61.1_linux_amd64 || exit
    nohup ./frps -c frps.toml > /dev/null 2>&1 &
)

#
#docker stop $(docker ps -a | grep "rec-flask" | awk '{print $1 }')
#docker rm $(docker ps -a | grep "rec-flask" | awk '{print $1 }')
#
#cd rec-flask && docker build . -t rec-flask
#
#docker run --name rec-flask -d -p 5000:5000  rec-flask
