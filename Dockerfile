FROM caddy as web
COPY . .
RUN echo ":8000" > /etc/caddy/Caddyfile
RUN echo "rewrite /repl /repl.html" >> /etc/caddy/Caddyfile
RUN echo "rewrite /compilerepl /compilerepl.html" >> /etc/caddy/Caddyfile
RUN echo "rewrite /cpsrepl /cpsrepl.html" >> /etc/caddy/Caddyfile
RUN echo "log" >> /etc/caddy/Caddyfile
RUN echo "file_server" >> /etc/caddy/Caddyfile

FROM alpine:latest as build
RUN printf -- '-m\nscrapscript\n...' > .args
RUN wget https://cosmo.zip/pub/cosmos/bin/assimilate
RUN wget https://cosmo.zip/pub/cosmos/bin/ape-x86_64.elf
RUN wget https://cosmo.zip/pub/cosmos/bin/python
RUN wget https://cosmo.zip/pub/cosmos/bin/zip
RUN chmod +x assimilate
RUN chmod +x ape-x86_64.elf
RUN chmod +x python
RUN chmod +x zip
RUN mkdir Lib
COPY scrapscript.py Lib/
COPY compiler.py Lib/
COPY runtime.c Lib/
COPY cli.c Lib/
RUN ./ape-x86_64.elf ./python -m compileall -b Lib/scrapscript.py Lib/compiler.py
RUN mv python scrapscript.com
RUN ./ape-x86_64.elf ./zip -r scrapscript.com Lib .args
RUN ./ape-x86_64.elf ./assimilate ./scrapscript.com
RUN echo "Testing..."
RUN ./scrapscript.com apply "1+2"

# Set up the container
FROM scratch as main
COPY --from=build scrapscript.com .
ENTRYPOINT ["./scrapscript.com"]
