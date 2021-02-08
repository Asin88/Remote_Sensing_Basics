var j = jQuery.noConflict();
j(document).ready(function() {
window.onload = function () {
setTimeout(function () {
	j("div.panel_button").toggle();
       j("div#panel").animate({
			height: "130px"
		});
}, 100);

}
	j("div.panel_button").click(function(){
		j("div#panel").animate({
			height: "130px"
		});		
		j("div.panel_button").toggle();
	
	});	
	
   j("div#hide_button").click(function(){
		j("div#panel").animate({
			height: "0px"
		}, "fast");
		
	
   });	
	
});