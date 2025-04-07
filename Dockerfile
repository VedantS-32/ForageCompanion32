FROM node:18 AS build-stage

WORKDIR /app

COPY package*.json tsconfig.json webpack.config.js ./
COPY static/ static/

RUN npm install
RUN npm run build


FROM python:3.9-slim

WORKDIR /app

COPY . .

COPY --from=build-stage /app/static/dist ./static/dist

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "entrypoint.py"]
