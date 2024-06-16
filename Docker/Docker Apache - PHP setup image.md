```DOCKERFILE

FROM buildpack-deps:stretch

RUN apt-get update && apt-get install -y curl nano vim libcrypt11-dev zlib1g-dev && rm -r /var/lib/apt/lists/*

##<apache2>##

RUN apt-get update && apt-get install -y apache2-bin apache2-dev apache2 --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN rm -rf /var/www/html && mkdir -p /var/lock/apache2 /var/run/apache2 /var/www/html && chown -R www-data:www-data /var/lock/apache2 /var/run/apache2 /var/log/apache2 /var/www/html

# Apache + PHP requires preforking Apache for best results
RUN a2dismod mpm_event && a2enmod mpm_prefork


RUN mv /etc/apache2/apache2.conf /etc/apache2/apache2.conf.dist
COPY apache2.conf /etc/apache2/apache2.conf
##</apache2>##

#RUN gpg --keyserver pgp.mit.edu --recv-keys 0B96609E270F565C13292B24C13C70B87267B52D 0A95E9A026542D53835E3F3A7DEC4E69FC9C83D7
ENV PHP_VERSION 5.3.29

# php 5.3 needs older autoconf
RUN set -x \
	&& apt-get update && apt-get install -y autoconf2.13 && rm -r /var/lib/apt/lists/* \
	&& autoconf --version \
	&& whereis autoconf2.13 \
	&& curl -SLO http://launchpadlibrarian.net/140087283/libbison-dev_2.7.1.dfsg-1_amd64.deb \
	&& curl -SLO http://launchpadlibrarian.net/140087282/bison_2.7.1.dfsg-1_amd64.deb \
	&& dpkg -i libbison-dev_2.7.1.dfsg-1_amd64.deb \
	&& dpkg -i bison_2.7.1.dfsg-1_amd64.deb \
	&& rm *.deb \
	&& curl -SL "http://php.net/get/php-$PHP_VERSION.tar.bz2/from/this/mirror" -o php.tar.bz2 \
	&& curl -SL "http://php.net/get/php-$PHP_VERSION.tar.bz2.asc/from/this/mirror" -o php.tar.bz2.asc
#	&& gpg --verify php.tar.bz2.asc \
ENV PHP_AUTOCONF /usr/bin/autoconf2.13

RUN curl -O https://www.openssl.org/source/old/1.0.2/openssl-1.0.2u.tar.gz \
	&& tar xf openssl-1.0.2u.tar.gz \
	&& cd openssl-1.0.2u \
	&& ./config -fPIC shared --prefix=/opt/openssl-1.0.2u \
	&& make \
	&& make install \
	&& cd ..

RUN mkdir -p /usr/src/php \
	&& tar -xf php.tar.bz2 -C /usr/src/php --strip-components=1 \
	&& rm php.tar.bz2* \
	&& cd /usr/src/php \
	&& ./buildconf --force \
	&& ./configure --disable-cgi \
		$(command -v apxs2 > /dev/null 2>&1 && echo '--with-apxs2' || true) \
		--with-mysql \
		--with-mysqli \
		--with-pdo-mysql \
		--with-openssl=/opt/openssl-1.0.2u \
		--with-zlib \
	&& make -j"$(nproc)" \
	&& make install \
	&& dpkg -r bison libbison-dev \
	&& apt-get purge -y --auto-remove autoconf2.13 \
	&& rm -r /usr/src/php

WORKDIR /var/www/html

EXPOSE 80
CMD ["apache2", "-DFOREGROUND"]
```

# apache2.conf 

```apache2.conf
# see http://sources.debian.net/src/apache2/2.4.10-1/debian/config-dir/apache2.conf

Mutex file:/var/lock/apache2 default
PidFile /var/run/apache2/apache2.pid
Timeout 300
KeepAlive On
MaxKeepAliveRequests 100
KeepAliveTimeout 5
User www-data
Group www-data
HostnameLookups Off
ErrorLog /proc/self/fd/2
LogLevel warn

IncludeOptional mods-enabled/*.load
IncludeOptional mods-enabled/*.conf

# ports.conf
Listen 80
<IfModule ssl_module>
	Listen 443
</IfModule>
<IfModule mod_gnutls.c>
	Listen 443
</IfModule>

<Directory />
	Options FollowSymLinks
	AllowOverride None
	Require all denied
</Directory>

<Directory /var/www/>
	Options Indexes FollowSymLinks
	AllowOverride All
	Require all granted
</Directory>

AccessFileName .htaccess
<FilesMatch "^\.ht">
	Require all denied
</FilesMatch>

LogFormat "%v:%p %h %l %u %t \"%r\" %>s %O \"%{Referer}i\" \"%{User-Agent}i\"" vhost_combined
LogFormat "%h %l %u %t \"%r\" %>s %O \"%{Referer}i\" \"%{User-Agent}i\"" combined
LogFormat "%h %l %u %t \"%r\" %>s %O" common
LogFormat "%{Referer}i -> %U" referer
LogFormat "%{User-agent}i" agent

CustomLog /proc/self/fd/1 combined

<FilesMatch \.php$>
	SetHandler application/x-httpd-php
</FilesMatch>
DirectoryIndex index.php

DocumentRoot /var/www/html
```
