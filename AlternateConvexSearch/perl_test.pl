use Config;
    
$ARCH = $Config{'archname'};    # Use perl's knowledge of the architecture.
print "My architecture is: ";
print $ARCH;
print "\nDone printing\n";