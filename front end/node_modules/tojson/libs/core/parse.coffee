Parser = (name, regExp, parser) ->
  @name = (if typeof name is "undefined" then "Default" else name)
  @regExp = null
  unless typeof regExp is "undefined"
    if typeof regExp is "string"
      @regExp = new RegExp(regExp)
    else
      @regExp = regExp
  @parse = parser  unless typeof parser is "undefined"
  return
module.exports = Parser
Parser::test = (str) ->
  unless @regExp?
    true
  else
    @regExp.test str


# Parser.prototype.newProcess=function(mixedColumnTitle){
#     var title=this.getTitle(mixedColumnTitle);
#     return {
#         "title"
#     }
# }
# Parser.prototype.getTitle=function(mixedTitle){
#     return mixedTitle.replace(this.regExp,"");
# }
Parser::parse = (params) ->
  params.resultRow[params.head] = params.item
  return

Parser::getHead = ->
  @head

Parser::clone = ->
  new Parser(@name, @regExp, @parse)

Parser::getName = ->
  @name

