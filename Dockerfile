# Set things up
FROM alpine:latest as build
ARG VER=3.0.2
ARG COSMO=cosmos-$VER.zip
RUN wget https://github.com/jart/cosmopolitan/releases/download/$VER/$COSMO
WORKDIR cosmo
RUN unzip ../$COSMO bin/ape.elf bin/assimilate bin/bash bin/python
RUN bin/ape.elf bin/assimilate bin/bash
RUN bin/ape.elf bin/assimilate bin/python

# Set up the container
FROM scratch
COPY scrapscript.py .
COPY --from=build /cosmo/bin/python .
EXPOSE 8000
ENTRYPOINT ["./python", "scrapscript.py"]
CMD ["serve", "--port", "8000"]
