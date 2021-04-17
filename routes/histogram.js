const express = require ('express')
const router = express.Router()
const path = require('path')


router.get('/age',(req,res)=>{
    res.sendFile(path.join(__dirname, '../images/histogram', 'Age.png'))
    
})
router.get('/fare',(req,res)=>{
    res.sendFile(path.join(__dirname, '../images/histogram', 'Fare.png'))
    
})
router.get('/parch',(req,res)=>{
    res.sendFile(path.join(__dirname, '../images/histogram', 'Parch.png'))
    
})
router.get('/pclass',(req,res)=>{
    res.sendFile(path.join(__dirname, '../images/histogram', 'Pclass.png'))
    
})
router.get('/sibsp',(req,res)=>{
    res.sendFile(path.join(__dirname, '../images/histogram', 'SibSp.png'))
    
})
router.get('/survived',(req,res)=>{
    res.sendFile(path.join(__dirname, '../images/histogram', 'Survived.png'))
    
})

module.exports = router