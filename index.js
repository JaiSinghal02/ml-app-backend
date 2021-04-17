const express = require('express')
const app = express()
const cors = require('cors')
const histogram = require('./routes/histogram')
const categorial = require('./routes/categorial')
const scatter = require('./routes/scatter')

var corsOptions = {
    origin: 'http://localhost:3000', //https://jaisinghal02.github.io
}


app.get('/',(req,res)=>{
    res.send("API for ML-App")
})
app.use(cors(corsOptions))
app.use('/image/histogram',histogram)
app.use('/image/categorial',categorial)
app.use('/image/scatter',scatter)
app.get('/python',(req,res)=>{
    const spawn = require("child_process").spawn
    const arr=[]
    Object.keys(req.query).forEach(q=>{
        arr.push(parseInt(req.query[q]))
    })
    var process = spawn('python',["./WebApp_test.py",arr] );
    let buf;
    process.stdout.on('data', function(data) {
    buf=Buffer.from(data)
    buf=JSON.parse(JSON.stringify(buf.toString()))
    res.send(buf)
    })
})


const port = process.env.PORT || 5000
app.listen(port, ()=>{
    console.log(`Server listeing on port ${port}`)
})