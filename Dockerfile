FROM caddy as web
COPY . .
RUN echo ":8000" > /etc/caddy/Caddyfile
RUN echo "rewrite /repl /repl.html" >> /etc/caddy/Caddyfile
RUN echo "rewrite /compilerepl /compilerepl.html" >> /etc/caddy/Caddyfile
RUN echo "log" >> /etc/caddy/Caddyfile
RUN echo "file_server" >> /etc/caddy/Caddyfile

FROM alpine:latest as build
RUN printf -- '-m\nscrapscript\n...' > .args
RUN wget https://cosmo.zip/pub/cosmos/bin/assimilate
RUN wget https://cosmo.zip/pub/cosmos/bin/python
RUN wget https://cosmo.zip/pub/cosmos/bin/zip
RUN chmod +x assimilate
RUN chmod +x python
RUN chmod +x zip
RUN mkdir Lib
COPY scrapscript.py Lib/
COPY compiler.py Lib/
COPY runtime.c Lib/
COPY cli.c Lib/
RUN sh -c "./python -m compileall -b Lib/scrapscript.py Lib/compiler.py"
RUN mv python scrapscript.com
RUN sh -c "./zip -r scrapscript.com Lib .args"
RUN echo "Testing..."
RUN sh -c './scrapscript.com apply "1+2"'
RUN sh -c "./assimilate scrapscript.com"

# Set up the container
FROM scratch as main
COPY --from=build scrapscript.com .
ENTRYPOINT ["./scrapscript.com"]
