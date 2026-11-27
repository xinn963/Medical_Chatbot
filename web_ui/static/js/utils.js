// Shared utilities for the Medical Chatbot interfaces

class ChatbotUtils {
    static formatMessage(text) {
        // Convert newlines to HTML breaks
        text = text.replace(/\n/g, '<br>');
        
        // Format confidence percentages
        text = text.replace(/\(([^)]*Confidence[^)]*)\)/g, '<span style="font-size: 12px; color: #6b7280; font-style: italic;">($1)</span>');
        
        // Format warnings and disclaimers
        text = text.replace(/⚠️([^.]*\.)/g, '<span style="color: #ef4444; font-weight: 500;">⚠️$1</span>');
        
        return text;
    }
    
    static addTypingEffect(element, text, speed = 30) {
        return new Promise((resolve) => {
            let i = 0;
            element.innerHTML = '';
            
            function typeWriter() {
                if (i < text.length) {
                    element.innerHTML += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, speed);
                } else {
                    resolve();
                }
            }
            
            typeWriter();
        });
    }
    
    static showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 9999;
            animation: slideInFromRight 0.3s ease-out;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        `;
        
        switch (type) {
            case 'success':
                notification.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
                break;
            case 'error':
                notification.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
                break;
            case 'warning':
                notification.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
                break;
            default:
                notification.style.background = 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)';
        }
        
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutToRight 0.3s ease-in forwards';
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }
    
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    static copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                this.showNotification('Copied to clipboard!', 'success', 2000);
            }).catch(() => {
                this.fallbackCopyTextToClipboard(text);
            });
        } else {
            this.fallbackCopyTextToClipboard(text);
        }
    }
    
    static fallbackCopyTextToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            this.showNotification('Copied to clipboard!', 'success', 2000);
        } catch (err) {
            this.showNotification('Failed to copy text', 'error', 2000);
        }
        
        document.body.removeChild(textArea);
    }
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInFromRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutToRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Make ChatbotUtils available globally
window.ChatbotUtils = ChatbotUtils;
