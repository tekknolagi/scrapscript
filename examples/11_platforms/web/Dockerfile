# Set things up
FROM alpine:latest as build
RUN apk add clang python3
ARG VER=3.4.0
RUN wget https://cosmo.zip/pub/cosmocc/cosmocc-$VER.zip
WORKDIR cosmo
RUN unzip ../cosmocc-$VER.zip
COPY scrapscript.py .
COPY compiler.py .
COPY runtime.c .
COPY examples/11_platforms/web/ .
RUN CC=bin/cosmocc python3 compiler.py --compile --platform web.c  handler.scrap
RUN bin/assimilate a.out
RUN du -sh a.out

# Set up the container
FROM scratch as server
COPY --from=build /cosmo/a.out .
EXPOSE 8000
ENTRYPOINT ["./a.out"]

# Deploy from the repo root with
# fly deploy --dockerfile examples/11_platforms/web/Dockerfile --local-only --config examples/11_platforms/web/fly.toml
