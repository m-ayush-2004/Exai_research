parserMgr = require("./parserMgr.js")
getType = require('./convertType.js').getType
getItem = require('./convertType.js').getItem



initDefaultParsers = ->
  i = 0

  while i < defaultParsers.length
    parserCfg = defaultParsers[i]
    parserMgr.addParser parserCfg.name, parserCfg.regExp, parserCfg.parserFunc
    i++
  return

_arrayParser = (params) ->
  type = getType params.head
  fieldName = params.head.replace(@regExp, "")
  params.resultRow[fieldName] = []  unless params.resultRow[fieldName]?
  params.resultRow[fieldName].push getItem type,params.item
  return

_jsonArrParser = (params) ->
  type = getType params.head
  fieldStr = params.head.replace(@regExp, "")
  headArr = fieldStr.split(".")
  pointer = params.resultRow
  while headArr.length > 1
    headStr = headArr.shift()
    pointer[headStr] = {}  unless pointer[headStr]?
    pointer = pointer[headStr]
  arrFieldName = headArr.shift()
  pointer[arrFieldName] = []  unless pointer[arrFieldName]?
  pointer[arrFieldName].push getItem type,params.item
  return

defaultParsers = [
  {
    name: "array"
    regExp: /^(#Number||#Boolean)?\*array\*/
    parserFunc: _arrayParser
  }
  {
    name: "json"
    regExp: /^(#Number||#Boolean)?\*json\*/
    parserFunc: require("./parser_json.js")
  }
  {
    name: "omit"
    regExp: /^\*omit\*/
    parserFunc: ->
  }
  {
    name: "jsonarray"
    regExp: /^(#Number||#Boolean)?\*jsonarray\*/
    parserFunc: _jsonArrParser
  }
]

initDefaultParsers()
