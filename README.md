# pont-server
The universal backend that serves our software-integrated plugins.

## Running locally
First build the container
```
docker built -t pont_tech_server .
```
And then run it
```
docker run -p 5000:5000 --gpus=all -d --restart always pont_tech_server
```

After that service could be accessed at localhost:5000
