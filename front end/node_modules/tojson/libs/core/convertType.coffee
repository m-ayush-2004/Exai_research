


exports.getType = getType = (str)->
  if str.indexOf('#Number') isnt -1
    return 'Number'
  else if str.indexOf('#Boolean') isnt -1
    return 'Boolean'
  null

exports.getItem = getItem = (type,item)->
	if type is 'Number'
		item = parseFloat item

	if type is 'Boolean'
		if item is 'True' or item is 'TRUE' or item is 'true'
			item = true
		else if item is 'False' or item is 'FALSE' or item is 'false'
			item = false

	return item