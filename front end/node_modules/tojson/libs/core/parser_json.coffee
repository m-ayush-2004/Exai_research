getType = require('./convertType.js').getType
getItem = require('./convertType.js').getItem

module.exports = (params) ->
  type = getType @getHead()
  fieldStr = @getHead().replace(@regExp, "")
  headArr = fieldStr.split(".")
  pointer = params.resultRow
  arrReg = /\[([0-9]*)\]/
  while headArr.length > 1 #go through all children
    headStr = headArr.shift()
    match = headStr.match(arrReg)
    if match #if its array, we need add an empty json object into specified index.
      pointer[headStr.replace(match[0], "")] = []  unless pointer[headStr.replace(match[0], "")]?
      index = match[1] #get index where json object should stay
      pointer = pointer[headStr.replace(match[0], "")]
      #if its dynamic array index, push to the end
      index = pointer.length  if index is ""
      #current index in the array is empty. we need create a new json object.
      pointer[index] = {}  unless pointer[index]
      pointer = pointer[index]
    else #not array, just normal JSON object. we get the reference of it
      pointer[headStr] = {}  unless pointer[headStr]?
      pointer = pointer[headStr]
  
  #now the pointer is pointing the position to add a key/value pair.
  key = headArr.shift()
  match = key.match(arrReg)
  if match # the last element is an array, we need check and treat it as an array.
    index = match[1]
    key = key.replace(match[0], "")
    pointer[key] = []  if not pointer[key] or not pointer[key] instanceof Array
    index = pointer[key].length  if index is ""
    pointer[key][index] = getItem type,params.item
  else #last element is normal
    pointer[key] = getItem type,params.item
  return
