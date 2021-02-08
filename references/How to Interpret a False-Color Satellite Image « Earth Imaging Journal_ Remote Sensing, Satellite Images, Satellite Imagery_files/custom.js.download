$ = jQuery.noConflict();

jQuery(function($){
$(document).ready(function(){

//SUPERFISH			
$("ul.sf-menu").supersubs({ 
	minWidth:    22,   
	maxWidth:    40,   
	extraWidth:  1     
	}).superfish({
	autoArrows: true,
	delay:  200, 
	speed:  'fast'
	}
);
 
//TICKER
$('#ticker-1').carouFredSel({
		width: 1000,
		align: false,
		items: {
			width: 'variable',
			height: 35,
			visible: 1
		},
		scroll: {
			delay: 1000,
			easing: 'linear',
			items: 1,
			duration: 0.07,
			pauseDuration: 0,
			pauseOnHover: 'immediate'
		}
	});
	
//	FRED
$('.caroufredsel_wrapper').css('width', '100%');
	
// CAROUSELS
$('#sections-photo').carousel({
	itemsPerPage: 2,
	itemsPerTransition: 2,
	easing: 'linear',
	noOfRows: 2
});

// CAROUSELS
$('#sections-video').carousel({
	itemsPerPage: 4,
	itemsPerTransition: 8,
	easing: 'linear',
	noOfRows: 2
});

$(".various").fancybox({
		maxWidth	: 800,
		maxHeight	: 600,
		fitToView	: false,
		width		: '70%',
		height		: '70%',
		autoSize	: false,
		closeClick	: false,
		openEffect	: 'none',
		closeEffect	: 'none'
	});
	
//SCROLLTOTOP
 $('a[href=#top]').click(function(){
        $('html, body').animate({scrollTop:0}, 'slow');
        return false;
    });
	
//FANCYBOX PHOTO GALLERY
$(".photo-gallery")
    .attr('rel', 'gallery1')
    .fancybox({
        beforeShow: function () {
            if (this.title) {
                // New line
                this.title += '<br />';
                
                // Add tweet button
                this.title += '<a href="https://twitter.com/share" class="twitter-share-button" data-count="none" data-url="' + this.href + '">Tweet</a> ';
                
                // Add FaceBook like button
                this.title += '<iframe src="//www.facebook.com/plugins/like.php?href=' + this.href + '&amp;layout=button_count&amp;show_faces=true&amp;width=500&amp;action=like&amp;font&amp;colorscheme=light&amp;height=23" scrolling="no" frameborder="0" style="border:none; overflow:hidden; width:110px; height:23px;" allowTransparency="true"></iframe>';
            }
        },
        afterShow: function() {
            // Render tweet button
            twttr.widgets.load();
        },
        helpers : {
            title : {
                type: 'inside'
            }
        }  
    });
    
//TOGGLE SHORTCODE	
$(".toggle_container").hide(); 

$("h3.trigger").click(function(){

$(this).toggleClass("active").next().slideToggle("normal");
	return false;
});

$(".container").fitVids();

}); // END doc ready
}); // END function