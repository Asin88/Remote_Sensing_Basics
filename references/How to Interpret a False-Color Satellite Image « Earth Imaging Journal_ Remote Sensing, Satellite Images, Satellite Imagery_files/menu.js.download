$ = jQuery.noConflict();

jQuery(function($){
$(document).ready(function(){

$("ul.sf-menu ul li").removeClass("current-menu-item");

//RESPONSIVE MENU
$("<select />").appendTo("#nav");
$("<option />", {
	"selected": "selected",
	"value"   : "",
	"text"    : "Menu"
}).appendTo("#nav select");

$("#nav a").each(function() {
	var el = $(this);

$("<option />", {
	"value"   : el.attr("href"),
	"text"    : el.text()
}).appendTo("#nav select");
});
		
//REMOVE OPTIONS WITH #
$("#nav option").each(function() {
var navOption = $(this);
if( navOption.val() == '#' ) {
navOption.remove();
}
});
		
$("#nav select").change(function() {
 window.location = $(this).find("option:selected").val();
});
		
//UNIFORM
$(function(){
 $("#nav select").uniform();
});	

}); // END doc ready
}); // END function