name = ArgMining
registry = hub.docker.com

build:
	docker build -t $(registry)/$(name) $(BUILD_OPTS) .

stop:
	docker rm -f $(name) || true

run: stop
	docker run -it --rm=true -v $(shell pwd):/var/www \
	--name=$(name) $(registry)/$(name) bash -l

start: stop
	docker run -d -v $(shell pwd):/var/www --name=$(name) $(registry)/$(name)
