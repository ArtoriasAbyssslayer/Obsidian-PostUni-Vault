
#asyncCode.js
```JS
  

/* Theory Event Driven Execution Callbacks Asynchronous (non-Blocking Code) */

// Sychronous code is a blocking code : This is because to execute code that is after the synchronous block should first the synchronous code get out of scope

// Asynchronous code allow slow operations that require processing to go into background and continue executing while other tasks after this code are able to start executing

// Asynchronous code utilizes callback system which means that when the work from the async task is done a callback function that is registered before is called to handle the result of the process

  
  
  

const fs = require('node:fs');

  

// Async function to read file

fs.readFile ('input.txt', 'utf-8', (err,data) =>{

    if(err){

        console.err("Error Reading the file" + err);

    }

    // This is where the callback returns to handle the readFile result

    console.log(data);

});

// Following line executes immediately even if the readfile has not finished

console.log('Reading File.....');

// After the readFile is done the callbacks returns to log the data.

  
  
  

/* Why async code is important in Javascript? */

/*

     JAVASCRIPT DESIGN

    Javascript is a single thread runtime/executions

    language on Node V8 engine (Only one thread for each applications)

  

    This means that all the users accessing the applications use the same thread

    AKA accessing the same process thread that is running on a computer which runs the application

  

    When one user blocks the single thread with sync code then all the other users have to wait

    for this code to be executed

  

    So if the other users have to do a simpler task (login, navigate, request resources,etc) can't do it

  

    (Heavy tasks are for async code - Async code is oftenly used in node.JS)

  

*/

  

// Non Blocking IO MODEL IS BASED ON ASYNC FUNCTIONS //

/* In other programming languages like PHP this works very differently because each user is creating a new

 Thread for each call. */
```
#binaryTree.js
```JS
class BTreeNode{

    constructor(val){

        this.val = val;

        this.left = null;

        this.right = null;

    }

}

  

class BinaryTree{

    constructor(){

        this.root = null;

    }

    // In order to insert root node

    insert(value) {

        this.root = this._insert(this.root, value);

      }

    _insert(node, value) {

    if (node === null) {

        return new BTreeNode(value);

    }

  

    if (value < node.val){

        node.left = this._insert(node.left,value);

    }else{

        node.right = this._insert(node.right, value);

    }

    return node;

    }

  

    // In-order traversal of the binary tree

    inOrderTraversal(callback){

        this._inOrderTraversal(this.root, callback);

    }

    _inOrderTraversal(node, callback){

        if(node !== null){

            this._inOrderTraversal(node.left,callback);

            callback(node.val);

            this._inOrderTraversal(node.right,callback);

        }

    }

    printTree() {

        this._printTree(this.root, 0);

      }

      _printTree(node, depth) {

        if (node !== null) {

          this._printTree(node.right, depth + 1);

          console.log('  '.repeat(depth) + node.val);

          this._printTree(node.left, depth + 1);

        }

      }

}

  
  

// Usage Example

  

const binTree = new BinaryTree();

binTree.insert(5);

binTree.insert(141);

binTree.insert(134);

binTree.insert(53);

binTree.insert(72);

binTree.insert(723);

binTree.insert(83);

  

// In-order traversal

const result = [];

binTree.inOrderTraversal(val => result.push(val));

console.log(result);

binTree.printTree();
```

#callbackFunctions.js
```JS
/* WHAT SHOULD NOT HAPPEN ON CALLBACK FUNCTIONS */

// The following code uses async readFile functions which one depends to another

// The code is obfuscated and can lead to uncontrollable results while a background readFile

// has not complete or most of the times halts the system like Sync Functions.

  

const fs = require('node:fs');

  

fs.readFile('start.txt', 'utf-8', (err,data1) =>{

    fs.readFile(`${data1}.txt`, 'utf-8',(err,data2) =>{

        fs.writeFile('final.txt', `${data2}, ${data1}`, 'utf-8', (err) =>{

            if(err) throw err;

            console.log('Your file has been writtten....');

        });

    });

});
```
#readWrite.js
```JS
const fs  = require('node:fs');

  
  

// create a server

// const server = http.createServer((req, res) => {

//     res.end('Hello from the server!');

// });

  

// server.listen(8000, '

//

// localhost', () => {

//     console.log('Listening to requests on port 8000');

// });

  

const textIn = fs.readFileSync('./starter/txt/input.txt', 'utf-8');

console.log(textIn);

const currentDate = new Date().toLocaleString();

// Use backticks (`) for template literals

const textOut = `What we know about avocado: ${textIn}.\n Created on ${currentDate}`;

console.log(textOut);

  

fs.writeFileSync('./starter/txt/output.txt', textOut);

const hello = "Hello World from Node.js";

console.log(hello);
```

#asyncCode 
```JS
const fs = require('fs');

const superagent = require('superagent');

  

const readFilePro = file => {

  return new Promise((resolve, reject) => {

    fs.readFile(file, (err, data) => {

      if (err) reject('I could not find that file 😢');

      resolve(data);

    });

  });

};

  

const writeFilePro = (file, data) => {

  return new Promise((resolve, reject) => {

    fs.writeFile(file, data, err => {

      if (err) reject('Could not write file 😢');

      resolve('success');

    });

  });

};

  

const getDogPic = async () => {

  try {

    const data = await readFilePro(`${__dirname}/dog.txt`);

    console.log(`Breed: ${data}`);

  

    const res1Pro = superagent.get(

      `https://dog.ceo/api/breed/${data}/images/random`

    );

    const res2Pro = superagent.get(

      `https://dog.ceo/api/breed/${data}/images/random`

    );

    const res3Pro = superagent.get(

      `https://dog.ceo/api/breed/${data}/images/random`

    );

    const all = await Promise.all([res1Pro, res2Pro, res3Pro]);

    const imgs = all.map(el => el.body.message);

    console.log(imgs);

  

    await writeFilePro('dog-img.txt', imgs.join('\n'));

    console.log('Random dog image saved to file!');

  } catch (err) {

    console.log(err);

  

    throw err;

  }

  return '2: READY 🐶';

};

  

(async () => {

  try {

    console.log('1: Will get dog pics!');

    const x = await getDogPic();

    console.log(x);

    console.log('3: Done getting dog pics!');

  } catch (err) {

    console.log('ERROR 💥');

  }

})();

  

/*

console.log('1: Will get dog pics!');

getDogPic()

  .then(x => {

    console.log(x);

    console.log('3: Done getting dog pics!');

  })

  .catch(err => {

    console.log('ERROR 💥');

  });

*/

/*

readFilePro(`${__dirname}/dog.txt`)

  .then(data => {

    console.log(`Breed: ${data}`);

    return superagent.get(`https://dog.ceo/api/breed/${data}/images/random`);

  })

  .then(res => {

    console.log(res.body.message);

    return writeFilePro('dog-img.txt', res.body.message);

  })

  .then(() => {

    console.log('Random dog image saved to file!');

  })

  .catch(err => {

    console.log(err);

  });

*/
```

#email.js
```JS
const nodemailer = require('nodemailer');

const pug = require('pug');

const htmlToText = require('html-to-text');

  

module.exports = class Email {

  constructor(user, url) {

    this.to = user.email;

    this.firstName = user.name.split(' ')[0];

    this.url = url;

    this.from = `Jonas Schmedtmann <${process.env.EMAIL_FROM}>`;

  }

  

  newTransport() {

    if (process.env.NODE_ENV === 'production') {

      // Sendgrid

      return nodemailer.createTransport({

        service: 'SendGrid',

        auth: {

          user: process.env.SENDGRID_USERNAME,

          pass: process.env.SENDGRID_PASSWORD

        }

      });

    }

  

    return nodemailer.createTransport({

      host: process.env.EMAIL_HOST,

      port: process.env.EMAIL_PORT,

      auth: {

        user: process.env.EMAIL_USERNAME,

        pass: process.env.EMAIL_PASSWORD

      }

    });

  }

  

  // Send the actual email

  async send(template, subject) {

    // 1) Render HTML based on a pug template

    const html = pug.renderFile(`${__dirname}/../views/email/${template}.pug`, {

      firstName: this.firstName,

      url: this.url,

      subject

    });

  

    // 2) Define email options

    const mailOptions = {

      from: this.from,

      to: this.to,

      subject,

      html,

      text: htmlToText.fromString(html)

    };

  

    // 3) Create a transport and send email

    await this.newTransport().sendMail(mailOptions);

  }

  

  async sendWelcome() {

    await this.send('welcome', 'Welcome to the Natours Family!');

  }

  

  async sendPasswordReset() {

    await this.send(

      'passwordReset',

      'Your password reset token (valid for only 10 minutes)'

    );

  }

};
```

#streamJS.js
```JS
const fs = require("fs");

const server = require("http").createServer();

  

server.on("request", (req, res) => {

  // Solution 1

  // fs.readFile("test-file.txt", (err, data) => {

  //   if (err) console.log(err);

  //   res.end(data);

  // });

  

  // Solution 2: Streams

  // const readable = fs.createReadStream("test-file.txt");

  // readable.on("data", chunk => {

  //   res.write(chunk);

  // });

  // readable.on("end", () => {

  //   res.end();

  // });

  // readable.on("error", err => {

  //   console.log(err);

  //   res.statusCode = 500;

  //   res.end("File not found!");

  // });

  

  // Solution 3

  const readable = fs.createReadStream("test-file.txt");

  readable.pipe(res);

  // readableSource.pipe(writeableDest)

});

  

server.listen(8000, "127.0.0.1", () => {

  console.log("Listening...");

});
```

#app.JS
```JS
const path = require('path');
const express = require('express');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const mongoSanitize = require('express-mongo-sanitize');
const xss = require('xss-clean');
const hpp = require('hpp');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
const compression = require('compression');
const cors = require('cors');

const AppError = require('./utils/appError');
const globalErrorHandler = require('./controllers/errorController');
const tourRouter = require('./routes/tourRoutes');
const userRouter = require('./routes/userRoutes');
const reviewRouter = require('./routes/reviewRoutes');
const bookingRouter = require('./routes/bookingRoutes');
const bookingController = require('./controllers/bookingController');
const viewRouter = require('./routes/viewRoutes');

// Start express app
const app = express();

app.enable('trust proxy');

app.set('view engine', 'pug');
app.set('views', path.join(__dirname, 'views'));

// 1) GLOBAL MIDDLEWARES
// Implement CORS
app.use(cors());
// Access-Control-Allow-Origin *
// api.natours.com, front-end natours.com
// app.use(cors({
//   origin: 'https://www.natours.com'
// }))

app.options('*', cors());
// app.options('/api/v1/tours/:id', cors());

// Serving static files
app.use(express.static(path.join(__dirname, 'public')));

// Set security HTTP headers
app.use(helmet());

// Development logging
if (process.env.NODE_ENV === 'development') {
  app.use(morgan('dev'));
}

// Limit requests from same API
const limiter = rateLimit({
  max: 100,
  windowMs: 60 * 60 * 1000,
  message: 'Too many requests from this IP, please try again in an hour!'
});
app.use('/api', limiter);

// Stripe webhook, BEFORE body-parser, because stripe needs the body as stream
app.post(
  '/webhook-checkout',
  bodyParser.raw({ type: 'application/json' }),
  bookingController.webhookCheckout
);

// Body parser, reading data from body into req.body
app.use(express.json({ limit: '10kb' }));
app.use(express.urlencoded({ extended: true, limit: '10kb' }));
app.use(cookieParser());

// Data sanitization against NoSQL query injection
app.use(mongoSanitize());

// Data sanitization against XSS
app.use(xss());

// Prevent parameter pollution
app.use(
  hpp({
    whitelist: [
      'duration',
      'ratingsQuantity',
      'ratingsAverage',
      'maxGroupSize',
      'difficulty',
      'price'
    ]
  })
);

app.use(compression());

// Test middleware
app.use((req, res, next) => {
  req.requestTime = new Date().toISOString();
  // console.log(req.cookies);
  next();
});

// 3) ROUTES
app.use('/', viewRouter);
app.use('/api/v1/tours', tourRouter);
app.use('/api/v1/users', userRouter);
app.use('/api/v1/reviews', reviewRouter);
app.use('/api/v1/bookings', bookingRouter);

app.all('*', (req, res, next) => {
  next(new AppError(`Can't find ${req.originalUrl} on this server!`, 404));
});

app.use(globalErrorHandler);

module.exports = app;

```
# Filesystem Tree
```
├───controllers  
├───dev-data  
│ ├───data  
│ ├───img  
│ └───templates  
├───models  
├───public  
│ ├───css  
│ ├───img  
│ │ ├───tours  
│ │ └───users  
│ └───js  
├───routes  
├───utils  
└───views  
└───email
```
![[Pasted image 20240616172545.png]]

#server
```JS
const mongoose = require('mongoose');

const dotenv = require('dotenv');

  

process.on('uncaughtException', err => {

  console.log('UNCAUGHT EXCEPTION! 💥 Shutting down...');

  console.log(err.name, err.message);

  process.exit(1);

});

  

dotenv.config({ path: './config.env' });

const app = require('./app');

  

const DB = process.env.DATABASE.replace(

  '<PASSWORD>',

  process.env.DATABASE_PASSWORD

);

  

mongoose

  .connect(DB, {

    useNewUrlParser: true,

    useCreateIndex: true,

    useFindAndModify: false

  })

  .then(() => console.log('DB connection successful!'));

  

const port = process.env.PORT || 3000;

const server = app.listen(port, () => {

  console.log(`App running on port ${port}...`);

});

  

process.on('unhandledRejection', err => {

  console.log('UNHANDLED REJECTION! 💥 Shutting down...');

  console.log(err.name, err.message);

  server.close(() => {

    process.exit(1);

  });

});

  

process.on('SIGTERM', () => {

  console.log('👋 SIGTERM RECEIVED. Shutting down gracefully');

  server.close(() => {

    console.log('💥 Process terminated!');

  });

});
```
