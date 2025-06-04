import { format, formatDistance, isToday, isYesterday } from 'date-fns';

export const formatMessageTime = (timestamp) => {
  const date = new Date(timestamp);
  
  if (isToday(date)) {
    return format(date, 'HH:mm');
  } else if (isYesterday(date)) {
    return `Yesterday ${format(date, 'HH:mm')}`;
  } else {
    return format(date, 'MMM dd, HH:mm');
  }
};

export const formatRelativeTime = (timestamp) => {
  return formatDistance(new Date(timestamp), new Date(), { addSuffix: true });
};